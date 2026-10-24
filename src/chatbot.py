"""
CineBot — the AI chatbot (Module 5) for CineMatch Pro.

A LangChain chatbot backed by Gemini, with a 10-turn conversation memory
and five tools that let it ground its answers in CineMatch Pro's own
recommendation engines rather than hallucinating movie facts:
``movie_search``, ``movie_details``, ``recommendation_tool``,
``sentiment_tool``, and ``trending_tool``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Callable

from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# NOTE: `langchain_google_genai`'s tool-schema conversion (as of the pinned
# 1.0.7) checks `issubclass(schema, BaseModel)` against LangChain's own
# `pydantic_v1` compatibility shim, not native pydantic v2. Tool arg
# schemas MUST subclass `langchain_core.pydantic_v1.BaseModel` (imported
# above) rather than `pydantic.BaseModel` directly, or `bind_tools()`
# raises "Unsupported function call type" at agent-construction time.

from src.config import GEMINI_API_KEY, GEMINI_MODEL, logger, require_api_key

if TYPE_CHECKING:  # type hints only — keeps PyTorch / ChromaDB out of the lite deployment
    from src.fusion_engine import FusionEngine
    from src.semantic_search import SemanticSearchEngine
    from src.sentiment_engine import SentimentEngine
    from src.tmdb_client import TMDBClient

_CINEBOT_PERSONA = (
    "You are CineBot, a knowledgeable and friendly movie critic friend "
    "inside the CineMatch Pro app. You speak casually and warmly, mixing "
    "English and Hindi naturally when the user does (Hinglish is fine). "
    "You LOVE movies and recommend thoughtfully, always explaining briefly "
    "why a movie fits what the user asked for. Use the tools available to "
    "you to ground your answers in real data — never invent ratings, cast, "
    "or plot details. Keep responses conversational and not too long. "
    "Whenever you recommend or mention a specific movie, write it exactly as "
    "**Title (Year)** in bold so the app can turn it into a clickable card."
)


_PLANNER_PROMPT = """You are the planning step of CineBot, a movie assistant.
Decide which tools (if any) are needed to answer the user's latest message.

Tools:
- movie_search(query: str): semantic search by description/mood/plot, e.g. "mind-bending thriller", or "movies like Interstellar".
- movie_details(title: str): overview, year and rating of ONE specific movie title.
- recommendation_tool(seed_movie_id: int): ONLY if a MovieLens id appeared earlier in the conversation.
- sentiment_tool(reviews: list[str]): only when the user pasted reviews.
- trending_tool(time_window: "day"|"week"): what is trending now.

Reply with ONLY a JSON object, no prose, in exactly this shape:
{"calls": [{"tool": "movie_search", "args": {"query": "..."}}]}
Use at most 3 calls. Use {"calls": []} for greetings or general chat."""

_MAX_TOOL_CALLS = 3
_MAX_OBSERVATION_CHARS = 1500


def _message_text(content: Any) -> str:
    """Flatten a LangChain message ``content`` (str or list of parts) to plain text.

    Newer Gemini models return a list of content parts instead of one string.

    Args:
        content: ``AIMessage.content`` — a string or a list of strings/dicts.

    Returns:
        str: The concatenated text.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def _parse_plan(text: str) -> list[dict[str, Any]]:
    """Extract the tool-call list from the planner's JSON reply.

    Args:
        text: Raw planner output (may be wrapped in markdown fences).

    Returns:
        list[dict[str, Any]]: Up to three ``{"tool", "args"}`` dicts; empty if
        the reply isn't valid JSON (the bot then simply chats without tools).
    """
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    calls = data.get("calls") if isinstance(data, dict) else None
    if not isinstance(calls, list):
        return []
    return [c for c in calls if isinstance(c, dict) and isinstance(c.get("tool"), str)][:_MAX_TOOL_CALLS]


class ChatbotError(RuntimeError):
    """Raised for CineBot-specific failures."""


# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------
class MovieSearchInput(BaseModel):
    """Input schema for the `movie_search` tool."""

    query: str = Field(description="Natural-language description of the kind of movie wanted.")


class MovieDetailsInput(BaseModel):
    """Input schema for the `movie_details` tool."""

    title: str = Field(description="Exact or approximate movie title to look up.")


class RecommendationInput(BaseModel):
    """Input schema for the `recommendation_tool`."""

    seed_movie_id: int = Field(description="MovieLens ID of a movie the user likes, used as the anchor.")
    user_id: int = Field(default=0, description="MovieLens user ID, if known; 0 for a guest/cold-start user.")


class SentimentInput(BaseModel):
    """Input schema for the `sentiment_tool`."""

    reviews: list[str] = Field(description="List of review texts to aggregate sentiment for.")


class TrendingInput(BaseModel):
    """Input schema for the `trending_tool`."""

    time_window: str = Field(default="week", description="'day' or 'week'.")


def _build_tools(
    semantic: SemanticSearchEngine | None,
    tmdb: TMDBClient,
    fusion: FusionEngine | None,
    sentiment: SentimentEngine | None,
    search_fn: Callable[[str], list[dict[str, Any]]] | None = None,
) -> list[StructuredTool]:
    """Wrap CineMatch Pro's engines as LangChain `StructuredTool`s.

    Args:
        semantic: Semantic search engine for `movie_search`.
        tmdb: TMDB client for `movie_details` and `trending_tool`.
        fusion: Fusion engine for `recommendation_tool`.
        sentiment: Sentiment engine for `sentiment_tool`.
        search_fn: Fallback for `movie_search` when there is no semantic
            engine (lite deployment): a callable returning movie dicts
            (``title`` and optionally ``movie_id`` / ``similarity_pct``).

    Returns:
        list[StructuredTool]: The five tools. A tool whose engine is not
        available answers that it is unavailable instead of failing.
    """

    def movie_search(query: str) -> str:
        """Semantic search over the movie corpus by natural-language query."""
        if semantic is not None:
            results = semantic.search(query, top_k=5)
        elif search_fn is not None:
            results = search_fn(query)
        else:
            return "Movie search is unavailable in this deployment."
        if not results:
            return "No matching movies found."
        return "; ".join(
            f"{r['title']}"
            + (f" [movie_id={r['movie_id']}]" if r.get("movie_id") is not None else "")
            + (f" ({r['similarity_pct']}% match)" if r.get("similarity_pct") is not None else "")
            for r in results
        )

    def movie_details(title: str) -> str:
        """Look up a movie's TMDB details by title."""
        results = tmdb.search_movies(title)
        if not results:
            return f"Could not find '{title}' on TMDB."
        top = results[0]
        return (
            f"{top.get('title')} ({(top.get('release_date') or '????')[:4]}): "
            f"{top.get('overview', 'No overview available.')} "
            f"TMDB rating: {top.get('vote_average', 'N/A')}/10."
        )

    def recommendation_tool(seed_movie_id: int, user_id: int = 0) -> str:
        """Get hybrid recommendations similar to a given movie."""
        if fusion is None:
            return "Recommendations are unavailable in this deployment."
        scored = fusion.recommend(user_id=user_id, seed_movie_id=seed_movie_id, top_k=5)
        if not scored:
            return "No recommendations found for that movie."
        def label(movie_id: int) -> str:
            try:
                return f"{fusion.movies.loc[movie_id, 'clean_title']} [movie_id={movie_id}]"
            except Exception:  # noqa: BLE001 - titles are a nicety
                return f"movie_id={movie_id}"

        return "; ".join(f"{label(m.movie_id)} (score={m.score})" for m in scored)

    def sentiment_tool(reviews: list[str]) -> str:
        """Aggregate positive-sentiment percentage across a list of reviews."""
        if sentiment is None:
            return "Sentiment analysis is unavailable in this deployment (it needs the RoBERTa model)."
        agg = sentiment.aggregate_movie_sentiment(reviews)
        return f"{agg['positive_pct']}% positive across {agg['review_count']} reviews."

    def trending_tool(time_window: str = "week") -> str:
        """Get currently trending movies from TMDB."""
        movies = tmdb.get_trending(time_window)
        if not movies:
            return "No trending data available right now."
        return "; ".join(m.get("title", "Unknown") for m in movies[:8])

    return [
        StructuredTool.from_function(
            func=movie_search,
            name="movie_search",
            description="Semantic search for movies matching a natural-language description, e.g. 'thriller with unreliable narrator'.",
            args_schema=MovieSearchInput,
        ),
        StructuredTool.from_function(
            func=movie_details,
            name="movie_details",
            description="Look up a specific movie's overview, year, and rating by title.",
            args_schema=MovieDetailsInput,
        ),
        StructuredTool.from_function(
            func=recommendation_tool,
            name="recommendation_tool",
            description="Get hybrid (collaborative + content + semantic) recommendations similar to a known movie ID.",
            args_schema=RecommendationInput,
        ),
        StructuredTool.from_function(
            func=sentiment_tool,
            name="sentiment_tool",
            description="Aggregate sentiment (percent positive) across a list of review texts.",
            args_schema=SentimentInput,
        ),
        StructuredTool.from_function(
            func=trending_tool,
            name="trending_tool",
            description="Get currently trending movies from TMDB, for the day or the week.",
            args_schema=TrendingInput,
        ),
    ]


class CineBot:
    """Conversational movie recommendation assistant (Module 5).

    Uses a transparent *plan → run tools → answer* loop on top of LangChain:
    Gemini first picks which of the five tools to call (as JSON), CineMatch
    Pro runs them locally, then Gemini writes the reply from the real results.
    Avoiding provider-native function calling keeps it working across Gemini
    model generations (Gemini 3 rejects tool calls whose "thought signature"
    isn't echoed back, which older LangChain agents cannot do).

    Attributes:
        tools: The five `StructuredTool`s the bot may use.
        memory: Rolling conversation window (``memory_turns`` exchanges).
    """

    def __init__(
        self,
        semantic: SemanticSearchEngine | None,
        tmdb: TMDBClient,
        fusion: FusionEngine | None,
        sentiment: SentimentEngine | None,
        api_key: str | None = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL,
        memory_turns: int = 10,
        search_fn: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        key = require_api_key("GEMINI_API_KEY", api_key)
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=0.7)
        self.tools = _build_tools(semantic, tmdb, fusion, sentiment, search_fn=search_fn)
        self._tools_by_name = {tool.name: tool for tool in self.tools}
        self.memory = ConversationBufferWindowMemory(
            k=memory_turns, memory_key="chat_history", return_messages=True
        )

    # -- internals -------------------------------------------------------
    def _history(self) -> list[Any]:
        return self.memory.load_memory_variables({}).get("chat_history", [])

    def _plan(self, message: str, history: list[Any]) -> list[dict[str, Any]]:
        """Ask Gemini which tools to call for this message."""
        reply = self.llm.invoke(
            [SystemMessage(content=_PLANNER_PROMPT), *history, HumanMessage(content=message)]
        )
        return _parse_plan(_message_text(reply.content))

    def _run_tools(self, calls: list[dict[str, Any]]) -> list[str]:
        """Execute planned tool calls locally; failures become readable observations."""
        observations: list[str] = []
        for call in calls:
            name, args = call["tool"], call.get("args") or {}
            tool = self._tools_by_name.get(name)
            if tool is None:
                continue
            try:
                result = str(tool.invoke(args))
            except Exception as exc:  # noqa: BLE001 - one failing tool must not sink the chat
                logger.warning("CineBot tool %s failed: %s", name, exc)
                result = f"(tool {name} failed: {exc})"
            observations.append(f"[{name}] {result[:_MAX_OBSERVATION_CHARS]}")
        return observations

    def _answer(self, message: str, history: list[Any], observations: list[str]) -> str:
        """Write the final reply grounded in the tool observations."""
        system = _CINEBOT_PERSONA
        if observations:
            system += "\n\nReal data from CineMatch Pro tools (use it, do not invent facts):\n" + "\n".join(observations)
        reply = self.llm.invoke([SystemMessage(content=system), *history, HumanMessage(content=message)])
        return _message_text(reply.content).strip()

    # -- public API ------------------------------------------------------
    def chat(self, message: str) -> str:
        """Send a user message to CineBot and get its reply.

        Args:
            message: The user's chat message.

        Returns:
            str: CineBot's response text.

        Raises:
            ChatbotError: If `message` is empty or the Gemini call fails.
        """
        if not message or not message.strip():
            raise ChatbotError("Cannot send an empty message to CineBot.")
        try:
            history = self._history()
            try:
                calls = self._plan(message, history)
            except Exception as exc:  # noqa: BLE001 - planning is optional; fall back to plain chat
                logger.warning("CineBot planning failed, answering without tools: %s", exc)
                calls = []
            reply = self._answer(message, history, self._run_tools(calls))
        except Exception as exc:  # noqa: BLE001 - surface as a domain error
            logger.error("CineBot invocation failed: %s", exc)
            raise ChatbotError(f"CineBot could not process that message: {exc}") from exc
        reply = reply or "Sorry, I couldn't come up with a response for that."
        self.memory.save_context({"input": message}, {"output": reply})
        return reply

    def reset(self) -> None:
        """Clear the conversation memory, starting a fresh session."""
        self.memory.clear()
        logger.info("CineBot conversation memory cleared.")


if __name__ == "__main__":
    logger.info(
        "CineBot requires GEMINI_API_KEY plus initialized SemanticSearchEngine, "
        "TMDBClient, FusionEngine, and SentimentEngine. See serving/api.py for wiring."
    )
