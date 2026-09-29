"""The single pipeline: embed the query, retrieve candidates, rerank them, ask the LLM
to pick and describe the best ones. app.py, chat.py and cli.py all call recommend() —
none has its own copy of this logic, so a change here reaches every interface at once."""

import cohere
import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient

from src.config import BOOKS_PATH, COLLECTION, MATCH_THRESHOLD, RERANK_TOP_N, RETRIEVE_LIMIT
from src.generate import generate
from src.rerank import get_rerank_client, rerank
from src.retrieve import search
from src.store import get_client


def recommend(
    query: str,
    qdrant_client: QdrantClient | None = None,
    openai_client: OpenAI | None = None,
    rerank_client: cohere.ClientV2 | None = None,
    books: pd.DataFrame | None = None,
    collection: str = COLLECTION,
    exclude_book_ids: set[str] | None = None,
) -> list[dict]:
    """Retrieve RETRIEVE_LIMIT books → rerank → keep the top RERANK_TOP_N → drop any
    below MATCH_THRESHOLD → let the LLM pick 3 to 5 and summarize them.

    Callers may pass in their own (cached) clients and books DataFrame; recommend(query)
    alone builds fresh ones, fine for one-shot CLI use but wasteful when called
    repeatedly (it would reconnect and reload books.jsonl every time).

    Returns dicts: rank, book_id, title, author, genres, match_score (the rerank score
    users see, 0 to 1, or None if reranking failed and we fell back to similarity
    order), similarity (cosine, kept for debugging), fit (the LLM's 1-10 judgment),
    brief_summary, matched_chunk. Empty list if nothing cleared the threshold."""
    qdrant_client = qdrant_client or get_client()
    openai_client = openai_client or OpenAI()
    if rerank_client is None:
        rerank_client = get_rerank_client()
    if books is None:
        books = pd.read_json(BOOKS_PATH, lines=True).set_index("book_id")

    # a refine request must not show the same books again; over-fetch by the number
    # excluded so the reranker still sees about RETRIEVE_LIMIT fresh candidates
    exclude = exclude_book_ids or set()
    retrieved = search(qdrant_client, openai_client, collection, query, limit=RETRIEVE_LIMIT + len(exclude))
    candidates = [
        {**r, "summary": books.loc[r["book_id"], "summary"]} for r in retrieved if r["book_id"] not in exclude
    ]

    reranked, ok = rerank(rerank_client, query, candidates, top_n=RERANK_TOP_N)
    if ok:
        # the threshold is on the rerank scale; in fallback mode there's no rerank
        # score to compare, so we keep the similarity-ordered top N instead
        reranked = [c for c in reranked if c["match_score"] >= MATCH_THRESHOLD]
    if not reranked:
        return []

    by_id = {c["book_id"]: c for c in reranked}
    generated = generate(openai_client, query, reranked)

    return [
        {
            "rank": rank,
            "book_id": g["book_id"],
            "title": g["title"],
            "author": g["author"],
            "genres": g["genres"],
            "match_score": by_id[g["book_id"]]["match_score"],
            "similarity": g["similarity"],
            "fit": g["fit"],
            "brief_summary": g["brief_summary"],
            "matched_chunk": by_id[g["book_id"]]["best_chunk_text"],
        }
        for rank, g in enumerate(generated, start=1)
    ]
