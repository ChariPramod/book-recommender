"""recommend() pipeline tests with search, rerank and generate replaced by fakes (no paid calls)."""

import pandas as pd

import src.recommend as rec
from src.config import MATCH_THRESHOLD


def candidate(book_id: str, similarity: float) -> dict:
    return {
        "book_id": book_id,
        "title": book_id.upper(),
        "author": f"author {book_id}",
        "genres": ["Fiction"],
        "similarity": similarity,
        "best_chunk_text": f"chunk {book_id}",
    }


BOOKS = pd.DataFrame(
    [{"book_id": b, "summary": f"summary {b}"} for b in ("a", "b", "c", "d")]
).set_index("book_id")


def run(monkeypatch, match_scores: dict, reranked_ok: bool, llm_order: list[str]) -> list[dict]:
    retrieved = [candidate(b, 0.5) for b in match_scores]
    monkeypatch.setattr(rec, "search", lambda *a, **k: retrieved)

    def fake_rerank(client, query, candidates, top_n):
        scored = [{**c, "match_score": match_scores[c["book_id"]] if reranked_ok else None} for c in candidates]
        if reranked_ok:
            scored.sort(key=lambda c: c["match_score"], reverse=True)
        return scored[:top_n], reranked_ok

    def fake_generate(client, query, candidates):
        by_id = {c["book_id"]: c for c in candidates}
        return [
            {**by_id[b], "fit": 7, "brief_summary": f"about {b}"} for b in llm_order if b in by_id
        ]

    monkeypatch.setattr(rec, "rerank", fake_rerank)
    monkeypatch.setattr(rec, "generate", fake_generate)
    return rec.recommend("q", qdrant_client=object(), openai_client=object(), rerank_client=object(), books=BOOKS)


def test_cards_sorted_by_match_score_not_llm_order(monkeypatch):
    results = run(monkeypatch, {"a": 0.9, "b": 0.8, "c": 0.7}, True, llm_order=["c", "a", "b"])
    assert [r["book_id"] for r in results] == ["a", "b", "c"]
    assert [r["rank"] for r in results] == [1, 2, 3]


def test_books_below_threshold_are_dropped(monkeypatch):
    below = MATCH_THRESHOLD - 0.01
    results = run(monkeypatch, {"a": 0.9, "b": below, "c": 0.7}, True, llm_order=["a", "b", "c"])
    assert [r["book_id"] for r in results] == ["a", "c"]


def test_nothing_above_threshold_returns_empty(monkeypatch):
    below = MATCH_THRESHOLD - 0.01
    assert run(monkeypatch, {"a": below, "b": below}, True, llm_order=["a", "b"]) == []


def test_fallback_keeps_llm_order_and_skips_threshold(monkeypatch):
    results = run(monkeypatch, {"a": 0.1, "b": 0.1, "c": 0.1}, False, llm_order=["c", "a"])
    assert [r["book_id"] for r in results] == ["c", "a"]
    assert all(r["match_score"] is None for r in results)
