"""Streamlit UI for the book recommender. Run with: streamlit run app.py"""

import pandas as pd
import streamlit as st
from openai import OpenAI

from src.config import BOOKS_PATH
from src.recommend import recommend
from src.store import get_client


@st.cache_resource
def get_qdrant_client():
    return get_client()


@st.cache_resource
def get_openai_client():
    return OpenAI()


@st.cache_resource
def get_books() -> pd.DataFrame:
    return pd.read_json(BOOKS_PATH, lines=True).set_index("book_id")


st.title("Book Recommender")

query = st.text_input("What are you in the mood to read?")
searched = st.button("Search")

if searched and query.strip():
    with st.spinner("Searching..."):
        results = recommend(
            query,
            qdrant_client=get_qdrant_client(),
            openai_client=get_openai_client(),
            books=get_books(),
        )

    if not results:
        st.write("No good matches found.")
    else:
        for result in results:
            with st.container(border=True):
                st.subheader(f"{result['rank']}. {result['title']}")
                st.caption(f"by {result['author']} — {', '.join(result['genres'])}")
                if result["match_score"] is not None:
                    st.write(f"match score: {result['match_score']:.2f}")
                else:
                    st.write("match score: unavailable (reranker offline)")
                st.write(f"LLM fit: {result['fit']}/10")
                st.write(result["brief_summary"])
                with st.expander("Matched passage"):
                    st.write(result["matched_chunk"])
                    st.caption(f"debug — embedding similarity: {result['similarity']:.3f}")
