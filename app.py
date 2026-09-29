"""Streamlit chat UI for the book recommender. Run with: streamlit run app.py"""

import streamlit as st

from src.chat import handle_message, load_resources, new_session
from src.config import MAX_MESSAGE_CHARS

GREETING = "Hi! Tell me what you're in the mood to read: a genre, a mood, a theme, or a book you loved."


@st.cache_resource
def get_resources() -> dict:
    # Qdrant client, OpenAI client, Cohere client and books.jsonl, built once per server process
    return load_resources()


def render_books(books: list[dict]) -> None:
    for book in books:
        with st.container(border=True):
            st.markdown(f"**{book['rank']}. {book['title']}**")
            st.caption(f"by {book['author']} · {', '.join(book['genres'])}")
            if book["match_score"] is not None:
                st.write(f"match score: {book['match_score']:.2f}")
            else:
                st.write("match score: unavailable (reranker offline)")
            st.write(book["brief_summary"])
            with st.expander("Why this book"):
                st.write(book["matched_chunk"])
                st.caption(f"LLM fit: {book['fit']}/10 · debug similarity: {book['similarity']:.3f}")


st.title("Book Recommender")

if "chat" not in st.session_state:
    st.session_state.chat = new_session()  # memory: last 10 turns + books shown
    st.session_state.messages = [{"role": "assistant", "text": GREETING, "books": []}]  # full display log

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["text"])
        render_books(message["books"])

if text := st.chat_input("Ask for a book recommendation", max_chars=MAX_MESSAGE_CHARS):
    st.session_state.messages.append({"role": "user", "text": text, "books": []})
    with st.chat_message("user"):
        st.write(text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = handle_message(st.session_state.chat, text, resources=get_resources())
        st.write(reply["text"])
        render_books(reply["books"])

    st.session_state.messages.append({"role": "assistant", "text": reply["text"], "books": reply["books"]})
