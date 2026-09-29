"""Chat logic tests with every paid call (router LLM, reply LLM, Qdrant, Cohere)
replaced by fakes, so they run offline and free."""

import pandas as pd
import pytest

import src.chat as chat
from src.config import MAX_MESSAGE_CHARS, MAX_SESSION_MESSAGES, MEMORY_TURNS


def fake_resources() -> dict:
    books = pd.DataFrame(
        [{"book_id": "b1", "title": "One", "author": "A", "genres": ["Fantasy"], "summary": "A wizard school."}]
    ).set_index("book_id")
    return {"qdrant": None, "openai": None, "rerank": None, "books": books}


def card(book_id: str, title: str) -> dict:
    return {"book_id": book_id, "title": title}


@pytest.fixture
def no_llm(monkeypatch):
    """Fail loudly if anything tries a real paid call."""

    def boom(*args, **kwargs):
        raise AssertionError("a paid call was made")

    monkeypatch.setattr(chat, "route", boom)
    monkeypatch.setattr(chat, "recommend", boom)
    monkeypatch.setattr(chat, "llm_reply", boom)
    monkeypatch.setattr(chat, "load_resources", boom)


def test_gate_blocks_long_message_before_any_llm_call(no_llm):
    session = chat.new_session()
    reply = chat.handle_message(session, "x" * (MAX_MESSAGE_CHARS + 1))
    assert reply["text"] == chat.TOO_LONG
    assert session["message_count"] == 0
    assert session["turns"] == []


def test_gate_allows_message_at_the_limit():
    assert chat.safety_gate(chat.new_session(), "x" * MAX_MESSAGE_CHARS) is None


def test_gate_blocks_after_session_limit(no_llm):
    session = chat.new_session()
    session["message_count"] = MAX_SESSION_MESSAGES
    reply = chat.handle_message(session, "recommend a mystery")
    assert reply["text"] == chat.SESSION_FULL


def test_memory_keeps_only_last_turns_but_all_shown_books():
    session = chat.new_session()
    for i in range(MEMORY_TURNS + 2):
        chat.remember(session, f"msg {i}", {"text": "ok", "books": [card(f"b{i}", f"T{i}")]})
    assert len(session["turns"]) == MEMORY_TURNS
    assert session["turns"][0]["user"] == "msg 2"
    assert len(session["shown_book_ids"]) == MEMORY_TURNS + 2
    assert session["last_results"] == [card(f"b{MEMORY_TURNS + 1}", f"T{MEMORY_TURNS + 1}")]


def test_history_numbers_shown_books_with_ids():
    turns = [{"user": "hi", "assistant": "here", "books": [card("b7", "Dune"), card("b9", "Emma")]}]
    history = chat.format_history(turns)
    assert "1. Dune [b7]; 2. Emma [b9]" in history


def test_refine_excludes_shown_books(monkeypatch):
    session = chat.new_session()
    session["shown_book_ids"] = ["b1", "b2"]
    captured = {}

    def fake_recommend(query, **kwargs):
        captured.update(query=query, exclude=kwargs["exclude_book_ids"])
        return []

    monkeypatch.setattr(chat, "route", lambda *a: {"intent": "refine", "search_query": "dark fantasy", "book_ref": None})
    monkeypatch.setattr(chat, "recommend", fake_recommend)
    reply = chat.handle_message(session, "something darker", resources=fake_resources())
    assert captured == {"query": "dark fantasy", "exclude": {"b1", "b2"}}
    assert "couldn't find strong matches" in reply["text"]


def test_refine_without_shown_books_becomes_recommend(monkeypatch):
    monkeypatch.setattr(chat, "route", lambda *a: {"intent": "refine", "search_query": "dark fantasy", "book_ref": None})
    monkeypatch.setattr(chat, "recommend", lambda query, **kwargs: [])
    reply = chat.handle_message(chat.new_session(), "something darker", resources=fake_resources())
    assert reply["intent"] == "recommend"


def test_about_unknown_book_asks_instead_of_guessing(monkeypatch):
    monkeypatch.setattr(chat, "route", lambda *a: {"intent": "about_shown_book", "search_query": "", "book_ref": "b999"})
    monkeypatch.setattr(chat, "llm_reply", lambda *a: "Which book do you mean?")
    reply = chat.handle_message(chat.new_session(), "tell me about it", resources=fake_resources())
    assert reply["intent"] == "unclear"
