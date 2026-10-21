"""
Sentiment intelligence engine (Module 3) for CineMatch Pro.

Fine-tunes a RoBERTa sequence-classification model on the IMDB 50k
review dataset and exposes inference helpers for scoring individual
reviews and aggregating per-movie sentiment dashboards.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import ROBERTA_MODEL_DIR, logger, settings

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class SentimentEngineError(RuntimeError):
    """Raised for sentiment-engine-specific failures."""


@dataclass
class SentimentResult:
    """Result of scoring a single review.

    Attributes:
        label: Predicted sentiment, ``"positive"`` or ``"negative"``.
        confidence: Softmax probability of the predicted label, in [0, 1].
    """

    label: str
    confidence: float


def _compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute accuracy/precision/recall/F1 for a Hugging Face `Trainer`.

    Args:
        eval_pred: ``(logits, labels)`` tuple as passed by `Trainer`.

    Returns:
        dict[str, float]: Metric name -> value.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class SentimentEngine:
    """RoBERTa-based binary sentiment classifier for movie reviews.

    Attributes:
        tokenizer: Hugging Face tokenizer matching the loaded/base model.
        model: Hugging Face sequence classification model.
        device: ``"cuda"`` if a GPU is available, else ``"cpu"``.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        source = str(model_path) if model_path else settings.sentiment.base_model
        logger.info("Loading sentiment model from '%s' onto device=%s", source, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        num_labels_kwargs = {}
        if not model_path:
            # Base checkpoint has its own label space; override for our
            # binary positive/negative task when starting from scratch.
            num_labels_kwargs = {
                "num_labels": 2,
                "id2label": ID2LABEL,
                "label2id": LABEL2ID,
                "ignore_mismatched_sizes": True,
            }
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, **num_labels_kwargs
        ).to(self.device)

    def _tokenize(self, batch: dict[str, list[str]]) -> dict[str, list[int]]:
        """Tokenize a batch of raw review texts.

        Args:
            batch: A batch dict with a ``"review"`` key mapping to a list
                of raw review strings (as produced by `datasets.Dataset.map`).

        Returns:
            dict[str, list[int]]: Tokenizer output (input_ids,
            attention_mask, etc.).
        """
        return self.tokenizer(
            batch["review"],
            truncation=True,
            padding="max_length",
            max_length=settings.sentiment.max_length,
        )

    def fine_tune(
        self,
        reviews_df: pd.DataFrame,
        output_dir: Path = ROBERTA_MODEL_DIR,
    ) -> dict[str, float]:
        """Fine-tune the model on the IMDB reviews dataset.

        Args:
            reviews_df: DataFrame with ``review`` (str) and ``sentiment``
                (``"positive"``/``"negative"``) columns.
            output_dir: Directory to save the fine-tuned model checkpoint.

        Returns:
            dict[str, float]: Evaluation metrics on the held-out test
            split (accuracy, precision, recall, f1).

        Raises:
            SentimentEngineError: If `reviews_df` is empty or malformed.
        """
        if reviews_df.empty:
            raise SentimentEngineError("Cannot fine-tune on empty reviews DataFrame.")
        if not {"review", "sentiment"}.issubset(reviews_df.columns):
            raise SentimentEngineError("reviews_df must contain 'review' and 'sentiment' columns.")

        df = reviews_df.copy()
        df["label"] = df["sentiment"].map(LABEL2ID)
        if df["label"].isna().any():
            raise SentimentEngineError("Found sentiment values outside {'positive','negative'}.")

        cfg = settings.sentiment
        train_df, test_df = train_test_split(
            df,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=df["label"],
        )
        logger.info("Fine-tuning on %d train / %d test reviews.", len(train_df), len(test_df))

        train_ds = HFDataset.from_pandas(train_df[["review", "label"]], preserve_index=False)
        test_ds = HFDataset.from_pandas(test_df[["review", "label"]], preserve_index=False)
        train_ds = train_ds.map(self._tokenize, batched=True)
        test_ds = test_ds.map(self._tokenize, batched=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,  # each RoBERTa checkpoint is ~500 MB
            fp16=torch.cuda.is_available(),  # ~2x faster on T4/V100/A100; invalid on CPU
            logging_steps=50,
            report_to=[],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=_compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()

        logger.info(
            "Fine-tuning complete. accuracy=%.4f f1=%.4f (target accuracy >%.2f)",
            metrics.get("eval_accuracy", 0.0),
            metrics.get("eval_f1", 0.0),
            cfg.target_accuracy,
        )

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # Trainer checkpoints (optimizer state etc.) are only useful for resuming;
        # leaving them inside the model folder bloats it by >1 GB.
        shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)
        logger.info("Saved fine-tuned sentiment model to %s", output_dir)

        return {
            "accuracy": metrics.get("eval_accuracy", 0.0),
            "precision": metrics.get("eval_precision", 0.0),
            "recall": metrics.get("eval_recall", 0.0),
            "f1": metrics.get("eval_f1", 0.0),
        }

    @torch.no_grad()
    def predict(self, text: str) -> SentimentResult:
        """Score a single review's sentiment.

        Args:
            text: Raw review text.

        Returns:
            SentimentResult: Predicted label and confidence.

        Raises:
            SentimentEngineError: If `text` is empty or whitespace only.
        """
        if not text or not text.strip():
            raise SentimentEngineError("Cannot score an empty review.")

        self.model.eval()
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=settings.sentiment.max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_id = int(torch.argmax(probs).item())
        label = self.model.config.id2label.get(pred_id, ID2LABEL.get(pred_id, "positive"))
        return SentimentResult(label=str(label).lower(), confidence=float(probs[pred_id].item()))

    @torch.no_grad()
    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
        """Score a batch of reviews efficiently.

        Args:
            texts: List of raw review texts.
            batch_size: Number of reviews to process per forward pass.

        Returns:
            list[SentimentResult]: One result per input text, same order.
        """
        self.model.eval()
        results: list[SentimentResult] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            inputs = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=settings.sentiment.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            for i, pred_id in enumerate(pred_ids.tolist()):
                label = self.model.config.id2label.get(pred_id, ID2LABEL.get(pred_id, "positive"))
                results.append(
                    SentimentResult(label=str(label).lower(), confidence=float(probs[i, pred_id].item()))
                )
        return results

    def aggregate_movie_sentiment(self, reviews: list[str]) -> dict[str, float]:
        """Compute an aggregate positive-sentiment percentage for a movie.

        Args:
            reviews: All review texts collected for one movie.

        Returns:
            dict[str, float]: ``{"positive_pct": ..., "review_count": ...}``.
            Returns ``positive_pct=0.0`` if `reviews` is empty.
        """
        if not reviews:
            return {"positive_pct": 0.0, "review_count": 0}
        results = self.predict_batch(reviews)
        positive_count = sum(1 for r in results if r.label == "positive")
        return {
            "positive_pct": round(100 * positive_count / len(results), 1),
            "review_count": len(results),
        }


if __name__ == "__main__":
    from src.data_loader import load_imdb_reviews

    reviews = load_imdb_reviews()
    engine = SentimentEngine()
    metrics = engine.fine_tune(reviews)
    logger.info("Final metrics: %s", metrics)
