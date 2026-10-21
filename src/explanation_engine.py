"""
LLM-powered explanation engine (part of Module 2D) for CineMatch Pro.

Generates a friendly, one-sentence "why we recommended this" explanation
using Gemini, given the seed movie and a candidate recommendation plus
its contributing signal scores from the fusion engine. Falls back to the
deterministic template in `fusion_engine.explain()` if the LLM call
fails, so recommendations never go unexplained.
"""

from __future__ import annotations

import google.generativeai as genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL, logger, require_api_key
from src.fusion_engine import FusionEngine, ScoredMovie

_SYSTEM_PROMPT = (
    "You are CineMatch Pro's recommendation-explanation writer. Given a "
    "seed movie a user just viewed, a recommended movie, and which "
    "recommendation signals drove the match, write ONE short, natural "
    "sentence (under 20 words) explaining why the recommendation fits. "
    "Be specific and conversational, not generic. Do not use quotation "
    "marks. Do not mention algorithm names like 'collaborative filtering'."
)


class ExplanationEngineError(RuntimeError):
    """Raised when the explanation engine cannot produce a response."""


class ExplanationEngine:
    """Generates natural-language "why recommended" explanations via Gemini.

    Attributes:
        model: Configured `genai.GenerativeModel` instance.
        fusion_engine: Optional `FusionEngine`, used for the deterministic
            fallback explanation if the LLM call fails.
    """

    def __init__(
        self,
        api_key: str | None = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL,
        fusion_engine: FusionEngine | None = None,
    ) -> None:
        key = require_api_key("GEMINI_API_KEY", api_key)
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name, system_instruction=_SYSTEM_PROMPT)
        self.fusion_engine = fusion_engine

    def explain(
        self,
        seed_title: str,
        recommended_title: str,
        scored_movie: ScoredMovie,
        extra_context: str = "",
    ) -> str:
        """Generate a one-sentence explanation for a single recommendation.

        Args:
            seed_title: Title of the movie the user was viewing.
            recommended_title: Title of the recommended movie.
            scored_movie: The `ScoredMovie` from `FusionEngine.recommend`,
                used to tell the LLM which signals drove the match.
            extra_context: Optional extra detail to ground the
                explanation, e.g. shared genre or director name.

        Returns:
            str: A short natural-language explanation. Falls back to a
            deterministic template (via `FusionEngine.explain`, if one
            was provided) on any LLM failure.
        """
        top_signal = (
            max(scored_movie.source_scores, key=scored_movie.source_scores.get)
            if scored_movie.source_scores
            else "content similarity"
        )
        prompt = (
            f"Seed movie: {seed_title}\n"
            f"Recommended movie: {recommended_title}\n"
            f"Strongest matching signal: {top_signal}\n"
            f"Extra context: {extra_context or 'none'}\n\n"
            "Write the one-sentence explanation now."
        )
        try:
            response = self.model.generate_content(prompt)
            text = (response.text or "").strip()
            if not text:
                raise ExplanationEngineError("Empty response from Gemini.")
            return text
        except Exception as exc:  # noqa: BLE001 - any LLM/SDK failure should degrade gracefully
            logger.warning("Explanation LLM call failed, using fallback: %s", exc)
            if self.fusion_engine is not None:
                return self.fusion_engine.explain(scored_movie, seed_title)
            return f"Recommended because it's similar to {seed_title}."


if __name__ == "__main__":
    logger.info(
        "ExplanationEngine requires GEMINI_API_KEY and, ideally, a FusionEngine "
        "for graceful fallback. See serving/api.py for wiring."
    )