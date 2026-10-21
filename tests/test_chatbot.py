"""Tests for src/chatbot.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.chatbot import CineBot, _build_tools
from src.fusion_engine import ScoredMovie


@pytest.fixture
def engine_mocks() -> dict[str, MagicMock]:
    """Mocked semantic/tmdb/fusion/sentiment engines for CineBot construction."""
    semantic = MagicMock()
    semantic.search.return_value = [{"title": "Memento", "similarity_pct": 91.0}]

    tmdb = MagicMock()
    tmdb.search_movies.return_value = [
        {"title": "Inception", "release_date": "2010-07-16", "overview": "A thief.", "vote_average": 8.4}
    ]
    tmdb.get_trending.return_value = [{"title": "RRR"}, {"title": "Pushpa"}]

    fusion = MagicMock()
    fusion.recommend.return_value = [ScoredMovie(movie_id=2, score=0.8, source_scores={})]

    sentiment = MagicMock()
    sentiment.aggregate_movie_sentiment.return_value = {"positive_pct": 87.0, "review_count": 10}

    return {"semantic": semantic, "tmdb": tmdb, "fusion": fusion, "sentiment": sentiment}


def test_build_tools_returns_five_tools(engine_mocks: dict[str, MagicMock]):
    """_build_tools produces exactly the 5 tools named in the product spec."""
    tools = _build_tools(**engine_mocks)
    names = {t.name for t in tools}
    assert names == {
        "movie_search",
        "movie_details",
        "recommendation_tool",
        "sentiment_tool",
        "trending_tool",
    }


def test_movie_search_tool_delegates_to_semantic_engine(engine_mocks: dict[str, MagicMock]):
    """The movie_search tool calls through to SemanticSearchEngine.search()."""
    tools = {t.name: t for t in _build_tools(**engine_mocks)}
    result = tools["movie_search"].func(query="mind-bending thriller")
    assert "Memento" in result


def test_sentiment_tool_formats_percentage(engine_mocks: dict[str, MagicMock]):
    """The sentiment_tool formats the aggregate positive percentage into its output string."""
    tools = {t.name: t for t in _build_tools(**engine_mocks)}
    result = tools["sentiment_tool"].func(reviews=["great movie", "terrible film"])
    assert "87.0%" in result


def test_trending_tool_lists_titles(engine_mocks: dict[str, MagicMock]):
    """The trending_tool lists trending movie titles from TMDB."""
    tools = {t.name: t for t in _build_tools(**engine_mocks)}
    result = tools["trending_tool"].func(time_window="week")
    assert "RRR" in result


def test_cinebot_constructs_with_ten_turn_memory(engine_mocks: dict[str, MagicMock]):
    """CineBot's memory window defaults to the spec's 10 conversation turns."""
    bot = CineBot(api_key="fake-key-for-construction-test", **engine_mocks)
    assert bot.memory.k == 10
    assert len(bot.tools) == 5


def test_cinebot_reset_clears_memory(engine_mocks: dict[str, MagicMock]):
    """reset() clears CineBot's conversation memory."""
    bot = CineBot(api_key="fake-key-for-construction-test", **engine_mocks)
    bot.memory.chat_memory.add_user_message("hello")
    assert len(bot.memory.chat_memory.messages) == 1
    bot.reset()
    assert len(bot.memory.chat_memory.messages) == 0


def _bot_with_fake_llm(engine_mocks, replies):
    """CineBot whose Gemini LLM is replaced by a scripted fake."""
    from langchain_core.messages import AIMessage

    bot = CineBot(api_key="fake-key-for-construction-test", **engine_mocks)
    scripted = iter(replies)
    seen_prompts = []

    class FakeLLM:
        def invoke(self, messages):
            seen_prompts.append(messages)
            return AIMessage(content=next(scripted))

    bot.llm = FakeLLM()
    return bot, seen_prompts


def test_chat_runs_planned_tool_and_grounds_the_answer(engine_mocks):
    """Plan -> run movie_search -> answer; the tool result reaches the answering prompt."""
    plan = '```json\n{"calls": [{"tool": "movie_search", "args": {"query": "twisty thriller"}}]}\n```'
    bot, prompts = _bot_with_fake_llm(engine_mocks, [plan, "Try Memento!"])
    assert bot.chat("suggest a twisty thriller") == "Try Memento!"
    engine_mocks["semantic"].search.assert_called_once()
    assert "Memento" in prompts[1][0].content
    assert len(bot.memory.chat_memory.messages) == 2


def test_chat_handles_list_content_from_newer_gemini_models(engine_mocks):
    """Gemini 3 returns content as a list of parts; the reply is flattened to text."""
    bot, _ = _bot_with_fake_llm(
        engine_mocks, ['{"calls": []}', [{"type": "text", "text": "Namaste! "}, {"type": "text", "text": "Kya dekhna hai?"}]]
    )
    assert bot.chat("hi") == "Namaste! Kya dekhna hai?"


def test_chat_survives_unparseable_plan_and_failing_tool(engine_mocks):
    """Garbage planner output means no tools; a crashing tool becomes an observation, not an error."""
    bot, _ = _bot_with_fake_llm(engine_mocks, ["not json at all", "Just chatting."])
    assert bot.chat("hello") == "Just chatting."

    engine_mocks["tmdb"].get_trending.side_effect = RuntimeError("TMDB down")
    plan = '{"calls": [{"tool": "trending_tool", "args": {"time_window": "week"}}]}'
    bot, prompts = _bot_with_fake_llm(engine_mocks, [plan, "Trending data is unavailable."])
    assert "unavailable" in bot.chat("what's trending?")
    assert "failed" in prompts[1][0].content


def test_chat_rejects_empty_message(engine_mocks):
    """Empty messages raise ChatbotError."""
    from src.chatbot import ChatbotError

    bot, _ = _bot_with_fake_llm(engine_mocks, [])
    with pytest.raises(ChatbotError):
        bot.chat("   ")
