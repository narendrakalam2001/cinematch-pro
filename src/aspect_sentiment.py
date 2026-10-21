"""
Aspect-based sentiment analysis (part of Module 3) for CineMatch Pro.

Breaks down per-movie sentiment into named aspects (story, acting,
direction, music, VFX) by combining spaCy sentence segmentation with
keyword matching against `ASPECT_KEYWORDS`, then scoring each matched
sentence with the fine-tuned `SentimentEngine`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import spacy
from spacy.language import Language

from src.config import ASPECT_KEYWORDS, logger
from src.sentiment_engine import SentimentEngine


class AspectSentimentError(RuntimeError):
    """Raised for aspect-sentiment-specific failures."""


@dataclass
class AspectScore:
    """Aggregated sentiment for a single aspect of one movie.

    Attributes:
        aspect: Aspect name, e.g. ``"story"``, ``"acting"``.
        positive_pct: Percentage of matched sentences classified positive.
        mention_count: Number of sentences mentioning this aspect.
        label: Human-readable quality label, e.g. ``"Excellent"``.
        stars: Star rating (1-5) derived from `positive_pct`.
    """

    aspect: str
    positive_pct: float
    mention_count: int
    label: str
    stars: int


@dataclass
class AspectSentimentReport:
    """Full aspect-based sentiment report for one movie.

    Attributes:
        overall_positive_pct: Overall positive-sentiment percentage
            across all reviews (not just aspect-matched sentences).
        aspects: Mapping of aspect name -> `AspectScore`.
    """

    overall_positive_pct: float
    aspects: dict[str, AspectScore] = field(default_factory=dict)


def _label_from_pct(positive_pct: float) -> tuple[str, int]:
    """Map a positive-sentiment percentage to a quality label and star count.

    Args:
        positive_pct: Percentage of positive sentiment, 0-100.

    Returns:
        tuple[str, int]: ``(label, stars)``, e.g. ``("Excellent", 5)``.
    """
    if positive_pct >= 90:
        return "Excellent", 5
    if positive_pct >= 75:
        return "Good", 4
    if positive_pct >= 60:
        return "Average", 3
    if positive_pct >= 40:
        return "Below Average", 2
    return "Poor", 1


_NLP_MODEL_NAME = "en_core_web_sm"


def _load_spacy_model() -> Language:
    """Load the spaCy English pipeline, with a helpful error if missing.

    Returns:
        spacy.language.Language: The loaded spaCy pipeline.

    Raises:
        AspectSentimentError: If the model is not installed.
    """
    try:
        return spacy.load(_NLP_MODEL_NAME)
    except OSError as exc:
        raise AspectSentimentError(
            f"spaCy model '{_NLP_MODEL_NAME}' not found. Install it with: "
            f"python -m spacy download {_NLP_MODEL_NAME}"
        ) from exc


class AspectSentimentAnalyzer:
    """Aspect-based sentiment analyzer for movie reviews.

    Attributes:
        nlp: spaCy pipeline used for sentence segmentation.
        sentiment_engine: `SentimentEngine` used to score matched
            sentences.
        aspect_keywords: Mapping of aspect name -> trigger keywords,
            sourced from `src.config.ASPECT_KEYWORDS`.
    """

    def __init__(
        self,
        sentiment_engine: SentimentEngine,
        aspect_keywords: dict[str, list[str]] = ASPECT_KEYWORDS,
    ) -> None:
        self.nlp = _load_spacy_model()
        self.sentiment_engine = sentiment_engine
        self.aspect_keywords = aspect_keywords

    def _extract_aspect_sentences(self, reviews: list[str]) -> dict[str, list[str]]:
        """Split reviews into sentences and bucket them by matched aspect.

        Args:
            reviews: Raw review texts for one movie.

        Returns:
            dict[str, list[str]]: Aspect name -> list of sentences that
            mention that aspect's keywords.
        """
        buckets: dict[str, list[str]] = defaultdict(list)
        docs = self.nlp.pipe(reviews, disable=["ner", "lemmatizer"])
        for doc in docs:
            for sent in doc.sents:
                sent_lower = sent.text.lower()
                for aspect, keywords in self.aspect_keywords.items():
                    if any(kw in sent_lower for kw in keywords):
                        buckets[aspect].append(sent.text.strip())
        return buckets

    def analyze(self, reviews: list[str]) -> AspectSentimentReport:
        """Produce a full aspect-based sentiment report for a movie.

        Args:
            reviews: All review texts collected for one movie.

        Returns:
            AspectSentimentReport: Overall sentiment plus a per-aspect
            breakdown (story, acting, direction, music, VFX). Aspects
            with zero mentions are omitted from `aspects`.

        Raises:
            AspectSentimentError: If `reviews` is empty.
        """
        if not reviews:
            raise AspectSentimentError("Cannot analyze sentiment for an empty review list.")

        overall = self.sentiment_engine.aggregate_movie_sentiment(reviews)
        aspect_buckets = self._extract_aspect_sentences(reviews)

        aspect_scores: dict[str, AspectScore] = {}
        for aspect, sentences in aspect_buckets.items():
            if not sentences:
                continue
            agg = self.sentiment_engine.aggregate_movie_sentiment(sentences)
            label, stars = _label_from_pct(agg["positive_pct"])
            aspect_scores[aspect] = AspectScore(
                aspect=aspect,
                positive_pct=agg["positive_pct"],
                mention_count=agg["review_count"],
                label=label,
                stars=stars,
            )

        logger.info(
            "Aspect sentiment analysis: overall=%.1f%%, aspects_found=%s",
            overall["positive_pct"],
            list(aspect_scores.keys()),
        )
        return AspectSentimentReport(
            overall_positive_pct=overall["positive_pct"], aspects=aspect_scores
        )

    def most_helpful_reviews(
        self, reviews: list[str], top_n: int = 1
    ) -> dict[str, list[str]]:
        """Pick the most confidently positive and negative reviews.

        Used to power the "Most helpful positive/negative review"
        highlights on the Movie Intelligence Card.

        Args:
            reviews: All review texts collected for one movie.
            top_n: Number of reviews to return per sentiment bucket.

        Returns:
            dict[str, list[str]]: ``{"positive": [...], "negative": [...]}``,
            each sorted by model confidence descending.

        Raises:
            AspectSentimentError: If `reviews` is empty.
        """
        if not reviews:
            raise AspectSentimentError("Cannot rank an empty review list.")

        results = self.sentiment_engine.predict_batch(reviews)
        scored = list(zip(reviews, results))

        positives = sorted(
            (r for r in scored if r[1].label == "positive"),
            key=lambda r: r[1].confidence,
            reverse=True,
        )[:top_n]
        negatives = sorted(
            (r for r in scored if r[1].label == "negative"),
            key=lambda r: r[1].confidence,
            reverse=True,
        )[:top_n]

        return {
            "positive": [text for text, _ in positives],
            "negative": [text for text, _ in negatives],
        }


if __name__ == "__main__":
    logger.info(
        "AspectSentimentAnalyzer requires a fine-tuned SentimentEngine. "
        "See src/sentiment_engine.py and scripts/train_sentiment.py."
    )
