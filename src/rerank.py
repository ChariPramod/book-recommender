"""RAG step 6: rerank retrieved books with Cohere Rerank.

Dense retrieval compares one query vector to one chunk vector, so a book can rank
high just because a single chunk shares words or a vibe with the query (e.g.
"The Romantics" for a romance query). A reranker reads the query and the whole
document text together and scores how well they actually match, on a 0 to 1 scale.

If Cohere is unavailable (no key, network error, API error), we fall back to the
similarity order with a warning instead of crashing — the app degrades, not dies.

Check before/after ordering for a query (one embedding call + one rerank call):
    python -m src.rerank "I want some novel recommendation related to romance"
"""

import sys
import warnings

import cohere
import pandas as pd

from src.config import (
    BOOKS_PATH,
    COHERE_API_KEY,
    COLLECTION,
    RERANK_MODEL,
    RERANK_SUMMARY_CHARS,
    RERANK_TOP_N,
    RETRIEVE_LIMIT,
)


def get_rerank_client() -> cohere.ClientV2 | None:
    if not COHERE_API_KEY:
        return None
    return cohere.ClientV2(api_key=COHERE_API_KEY)


def build_document(candidate: dict) -> str:
    # "title. genres. summary" — title and genres give the reranker the same book
    # context the chunk prefix gave the embedder; the summary is trimmed so every
    # document stays well under the reranker's per-document token limit
    genres = ", ".join(candidate["genres"])
    summary = candidate["summary"][:RERANK_SUMMARY_CHARS]
    return f"{candidate['title']}. {genres}. {summary}"


def rerank(
    client: cohere.ClientV2 | None,
    query: str,
    candidates: list[dict],
    top_n: int = RERANK_TOP_N,
) -> tuple[list[dict], bool]:
    """candidates: dicts with at least title, genres, summary, similarity.

    Returns (top_n candidates, reranked). Each returned candidate is a copy with a
    "match_score" key: the rerank relevance score (0 to 1) when reranked is True,
    or None when we fell back to similarity order (cosine scores aren't on the
    same scale, so they must not be compared against MATCH_THRESHOLD)."""
    if not candidates:
        return [], True

    fallback = [{**c, "match_score": None} for c in candidates[:top_n]]
    if client is None:
        warnings.warn("COHERE_API_KEY not set; showing results in similarity order without reranking")
        return fallback, False

    try:
        response = client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[build_document(c) for c in candidates],
            top_n=top_n,
        )
    except Exception as error:  # any Cohere/network failure must not take the app down
        warnings.warn(f"Cohere rerank failed ({type(error).__name__}); showing results in similarity order")
        return fallback, False

    return [
        {**candidates[result.index], "match_score": result.relevance_score}
        for result in response.results
    ], True


def main(query: str) -> None:
    from openai import OpenAI

    from src.retrieve import search
    from src.store import get_client

    books = pd.read_json(BOOKS_PATH, lines=True).set_index("book_id")
    retrieved = search(get_client(), OpenAI(), COLLECTION, query, limit=RETRIEVE_LIMIT)
    candidates = [{**r, "summary": books.loc[r["book_id"], "summary"]} for r in retrieved]

    print(f"query: {query}\n")
    print(f"BEFORE reranking: top {RERANK_TOP_N} of {len(candidates)} by similarity")
    for i, c in enumerate(candidates[:RERANK_TOP_N], start=1):
        print(f"  {i:2}. similarity {c['similarity']:.3f}  {c['title']} by {c['author']}")

    reranked, ok = rerank(get_rerank_client(), query, candidates)
    label = "match" if ok else "match n/a (fallback)"
    print(f"\nAFTER reranking ({RERANK_MODEL}): top {len(reranked)}")
    for i, c in enumerate(reranked, start=1):
        match = f"{c['match_score']:.3f}" if c["match_score"] is not None else "  -  "
        print(f"  {i:2}. {label} {match}  similarity {c['similarity']:.3f}  {c['title']} by {c['author']}")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
