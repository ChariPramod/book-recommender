"""Conversation manager: safety gate, session memory, and one handler per intent.

handle_message(session, text) -> {"text", "books", "intent", "search_query"}
"books" holds result cards (only for recommend / refine); "intent" and
"search_query" are there for logging and the smoke test.

The session is a plain dict (see new_session), so the Streamlit app can keep it in
st.session_state and the smoke test can keep it in a local variable.
"""

import logging
from functools import lru_cache

import pandas as pd
from openai import OpenAI

from src.config import (
    ABOUT_BOOK_SUMMARY_CHARS,
    BOOKS_PATH,
    LLM_MODEL,
    MAX_MESSAGE_CHARS,
    MAX_SESSION_MESSAGES,
    MEMORY_TURNS,
)
from src.recommend import recommend
from src.rerank import get_rerank_client
from src.router import route
from src.store import get_client

logger = logging.getLogger("chat")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[chat] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

TOO_LONG = f"That message is a bit long. Please keep it under {MAX_MESSAGE_CHARS} characters."
SESSION_FULL = (
    f"This conversation has reached its limit of {MAX_SESSION_MESSAGES} messages. "
    "Refresh the page to start a new one."
)


def new_session() -> dict:
    return {
        "message_count": 0,  # user messages accepted by the safety gate
        "turns": [],  # last MEMORY_TURNS turns: {"user", "assistant", "books": [{"book_id", "title"}]}
        "last_results": [],  # full cards from the most recent recommendation
        "shown_book_ids": [],  # every book shown this session, so refine can exclude them
    }


@lru_cache(maxsize=1)
def load_resources() -> dict:
    # built once per process; app.py also wraps this in st.cache_resource
    return {
        "qdrant": get_client(),
        "openai": OpenAI(),
        "rerank": get_rerank_client(),
        "books": pd.read_json(BOOKS_PATH, lines=True).set_index("book_id"),
    }


def safety_gate(session: dict, text: str) -> str | None:
    """Runs before any LLM call. Returns a limit message if the message must be
    blocked, or None if it may go through."""
    if len(text) > MAX_MESSAGE_CHARS:
        return TOO_LONG
    if session["message_count"] >= MAX_SESSION_MESSAGES:
        return SESSION_FULL
    return None


def format_history(turns: list[dict]) -> str:
    """Compact history for the LLM: user text, assistant text, and the books shown
    as numbered "title [book_id]" so the router can resolve "the second one"."""
    if not turns:
        return "(no earlier messages)"
    lines = []
    for turn in turns:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
        if turn["books"]:
            shown = "; ".join(f"{i}. {b['title']} [{b['book_id']}]" for i, b in enumerate(turn["books"], start=1))
            lines.append(f"Books shown: {shown}")
    return "\n".join(lines)


def remember(session: dict, user_text: str, reply: dict) -> None:
    books = [{"book_id": b["book_id"], "title": b["title"]} for b in reply["books"]]
    session["turns"].append({"user": user_text, "assistant": reply["text"], "books": books})
    session["turns"] = session["turns"][-MEMORY_TURNS:]
    if reply["books"]:
        session["last_results"] = reply["books"]
        for book in books:
            if book["book_id"] not in session["shown_book_ids"]:
                session["shown_book_ids"].append(book["book_id"])


def llm_reply(client: OpenAI, instruction: str, history_text: str, text: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a friendly assistant in a book recommendation chat. Keep replies to 1 to 3 "
                "sentences. Never name or recommend specific books in this reply: recommendations only come "
                f"from the search system.\n\n{instruction}",
            },
            {"role": "user", "content": f"Conversation so far:\n{history_text}\n\nUser: {text}"},
        ],
    )
    return response.choices[0].message.content.strip()


def answer_about_book(client: OpenAI, book: pd.Series, text: str) -> str:
    summary = book["summary"][:ABOUT_BOOK_SUMMARY_CHARS]
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer the user's question about one book using ONLY the stored summary below. "
                "Don't use outside knowledge about this book or its author. If the summary doesn't answer the "
                "question, say the stored summary doesn't cover it. Keep it to 2 to 4 sentences.\n\n"
                f"Title: {book['title']}\nAuthor: {book['author']}\n"
                f"Genres: {', '.join(book['genres'])}\nSummary: {summary}",
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content.strip()


def handle_message(session: dict, text: str, resources: dict | None = None) -> dict:
    text = text.strip()
    blocked = safety_gate(session, text)
    if blocked:
        logger.info(f"blocked: {blocked}")
        return {"text": blocked, "books": [], "intent": "blocked", "search_query": ""}
    if not text:
        return {"text": "Tell me what you're in the mood to read.", "books": [], "intent": "empty", "search_query": ""}

    session["message_count"] += 1
    res = resources or load_resources()
    history_text = format_history(session["turns"])

    routed = route(res["openai"], text, history_text)
    logger.info(f"router raw: {routed}")  # before our corrections, so a downgrade is traceable
    intent, search_query = routed["intent"], routed["search_query"].strip()
    # the history shows ids as "[b01975]" and the router has copied the brackets
    # before (smoke test, 2026-09-29), which made every follow-up look unresolvable
    book_ref = (routed["book_ref"] or "").strip().strip("[]").strip() or None

    shown_ids = set(session["shown_book_ids"])
    if intent == "refine" and not shown_ids:
        intent = "recommend"  # nothing to refine yet
    if intent == "about_shown_book" and book_ref not in shown_ids:
        intent = "unclear"  # the router couldn't pin down a shown book; ask instead of guessing
    if intent in ("recommend", "refine") and not search_query:
        search_query = text

    books: list[dict] = []
    if intent in ("recommend", "refine"):
        books = recommend(
            search_query,
            qdrant_client=res["qdrant"],
            openai_client=res["openai"],
            rerank_client=res["rerank"],
            books=res["books"],
            exclude_book_ids=shown_ids if intent == "refine" else None,
        )
        if books:
            reply_text = f'Here are some books for "{search_query}":'
        else:
            reply_text = (
                f'I couldn\'t find strong matches for "{search_query}". Try naming a genre, a theme, '
                "a setting, or a book you enjoyed."
            )
    elif intent == "about_shown_book":
        reply_text = answer_about_book(res["openai"], res["books"].loc[book_ref], text)
    elif intent == "small_talk":
        reply_text = llm_reply(
            res["openai"],
            "Reply warmly to the small talk, then invite them to share what they'd like to read "
            "(a genre, mood, theme, or a book they loved).",
            history_text,
            text,
        )
    elif intent == "unclear":
        reply_text = llm_reply(
            res["openai"],
            "The user's request is too vague to search. Ask exactly ONE short clarifying question "
            "about what they want to read.",
            history_text,
            text,
        )
    else:  # off_topic
        reply_text = llm_reply(
            res["openai"],
            "The message is off topic. Politely say you can only help with book recommendations and "
            "invite them to ask for one. Don't answer the off-topic question.",
            history_text,
            text,
        )

    reply = {"text": reply_text, "books": books, "intent": intent, "search_query": search_query}
    remember(session, text, reply)
    logger.info(f"intent={intent} query={search_query!r} books={[b['title'] for b in books]}")
    return reply
