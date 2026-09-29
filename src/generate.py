"""RAG step 7: pick and describe 3 to 5 books from the retrieved candidates.

The LLM is trusted to pick and to summarize — never to report facts. title,
author, genres and similarity always come from our own data (the candidates
passed in), never from the LLM's output, and any book_id it returns that
isn't one of our candidates is dropped rather than trusted.
"""

import json

from openai import OpenAI

from src.config import LLM_MODEL

MAX_SUMMARY_CHARS = 1500

SYSTEM_PROMPT = """You are a book recommender.

You are given a user's request and a numbered list of candidate books, each with a
book_id, title, author, genres, and summary. Recommend ONLY books from this candidate
list — never invent a book or use any outside knowledge about these books or others.

Pick the 3 to 5 candidates that best match the request, ordered best match first. For
each one, write a brief 1 to 2 sentence summary based ONLY on the summary text given for
that candidate, and a fit score from 1 (barely fits) to 10 (excellent fit) representing
your own judgment of how well the book matches the request. If none of the candidates
are a good fit for the request, return an empty list rather than forcing a match.

Respond with JSON only, in exactly this shape:
{"recommendations": [{"book_id": "...", "brief_summary": "...", "fit": 1}]}
"""

# a strict JSON schema, not just response_format={"type": "json_object"} — that
# looser mode only guarantees valid JSON, not that the model uses our exact key
# names, so brief_summary can come back under a different key and silently
# render blank. Strict mode forces the model to use exactly these keys.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "book_recommendations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "book_id": {"type": "string"},
                            "brief_summary": {"type": "string"},
                            "fit": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["book_id", "brief_summary", "fit"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["recommendations"],
            "additionalProperties": False,
        },
    },
}


def build_candidates_text(candidates: list[dict]) -> str:
    blocks = []
    for candidate in candidates:
        summary = candidate["summary"][:MAX_SUMMARY_CHARS]
        genres = ", ".join(candidate["genres"])
        blocks.append(
            f"book_id: {candidate['book_id']}\n"
            f"title: {candidate['title']}\n"
            f"author: {candidate['author']}\n"
            f"genres: {genres}\n"
            f"summary: {summary}"
        )
    return "\n\n".join(blocks)


def generate(client: OpenAI, query: str, candidates: list[dict]) -> list[dict]:
    """candidates: dicts with book_id, title, author, genres, similarity, summary
    (summary comes from books.jsonl, not from a chunk — the LLM sees the whole
    book's summary, not just the piece that was retrieved).

    Returns dicts with book_id, title, author, genres, similarity, brief_summary, fit
    (the LLM's own 1-10 judgment, distinct from similarity, which is our cosine score),
    in the order the LLM picked them, 3 to 5 long (or empty if nothing fit)."""
    candidates_by_id = {c["book_id"]: c for c in candidates}
    candidates_text = build_candidates_text(candidates)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        response_format=RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Request: {query}\n\nCandidates:\n{candidates_text}"},
        ],
    )
    parsed = json.loads(response.choices[0].message.content)
    raw_recommendations = parsed.get("recommendations", [])

    results = []
    seen_ids: set[str] = set()
    for rec in raw_recommendations:
        book_id = rec.get("book_id")
        # only trust book_ids we actually offered as candidates, and only once each
        if book_id not in candidates_by_id or book_id in seen_ids:
            continue
        seen_ids.add(book_id)
        candidate = candidates_by_id[book_id]
        results.append(
            {
                "book_id": book_id,
                "title": candidate["title"],
                "author": candidate["author"],
                "genres": candidate["genres"],
                "similarity": candidate["similarity"],
                "brief_summary": rec.get("brief_summary", ""),
                "fit": rec.get("fit"),
            }
        )
    return results
