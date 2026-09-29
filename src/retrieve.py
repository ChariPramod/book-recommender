"""RAG step 5: dense retrieval — embed a query, find the best-matching books.

Scope for today: dense only. No BM25, no hybrid, no reranker, no eval — those come later.
"""

from openai import OpenAI
from qdrant_client import QdrantClient

from src.embed import embed_texts

MAX_BOOKS_PER_AUTHOR = 2


def search(
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    collection: str,
    query: str,
    limit: int = 30,
) -> list[dict]:
    """Embeds `query` with the same model used for the chunks, then asks Qdrant
    for the best-matching book for each of up to `limit` books — grouping by
    book_id (group_size=1) means each book is represented by its single best
    chunk, not by however many chunks it happens to have.

    Caps results at MAX_BOOKS_PER_AUTHOR per author afterward, so one prolific
    author can't crowd out everyone else, while keeping the score order.
    """
    query_vector = embed_texts(openai_client, [query])[0]

    groups = qdrant_client.query_points_groups(
        collection_name=collection,
        query=query_vector.tolist(),
        group_by="book_id",
        group_size=1,
        limit=limit,
    ).groups

    results = []
    for group in groups:
        hit = group.hits[0]  # group_size=1, so there's exactly one hit per group
        results.append(
            {
                "book_id": hit.payload["book_id"],
                "title": hit.payload["title"],
                "author": hit.payload["author"],
                "genres": hit.payload["genres"],
                "similarity": hit.score,
                "best_chunk_text": hit.payload["raw_text"],
            }
        )

    books_per_author: dict[str, int] = {}
    capped = []
    for result in results:
        author = result["author"]
        if books_per_author.get(author, 0) >= MAX_BOOKS_PER_AUTHOR:
            continue
        books_per_author[author] = books_per_author.get(author, 0) + 1
        capped.append(result)

    return capped
