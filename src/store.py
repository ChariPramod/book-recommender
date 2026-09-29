"""RAG step 4: store chunk vectors in Qdrant Cloud.

Usage:
    python -m src.store data/processed/chunks_sentence_230_prefix_ov1.jsonl \\
        data/processed/embeddings_sentence_230_prefix_ov1.npy
"""

import sys
import uuid

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import BOOKS_PATH, QDRANT_API_KEY, QDRANT_URL, collection_name

# fixed, so uuid5(chunk_id) is the same point id every run (re-running never
# creates duplicate points, it just overwrites the same ones)
CHUNK_NAMESPACE = uuid.UUID("c9b1f3d2-6b8a-4c3f-8e2a-1a6f9d5c7b40")


def get_client(url: str = QDRANT_URL, api_key: str = QDRANT_API_KEY) -> QdrantClient:
    # the API key authenticates us to the cloud cluster; gRPC sends vectors as raw
    # bytes instead of JSON text, which is 2 to 3 times less data to upload
    return QdrantClient(url=url, api_key=api_key, prefer_grpc=True, timeout=120)


def create_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def create_payload_indexes(client: QdrantClient, name: str) -> None:
    # Qdrant's docs recommend creating indexes before ingest, so they're built
    # incrementally as points arrive instead of over the whole collection at once
    for field in ("book_id", "author", "genres"):
        client.create_payload_index(name, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD)


def build_payloads(chunks: pd.DataFrame, books: pd.DataFrame) -> list[dict]:
    payloads = []
    for chunk in chunks.itertuples():
        book = books.loc[chunk.book_id]
        payloads.append(
            {
                "chunk_id": chunk.chunk_id,
                "book_id": chunk.book_id,
                "chunk_index": int(chunk.chunk_index),
                "raw_text": chunk.raw_text,
                "n_tokens": int(chunk.n_tokens),
                "title": book["title"],
                "author": book["author"],
                "genres": book["genres"],
            }
        )
    return payloads


def build_ids(chunks: pd.DataFrame) -> list[str]:
    return [str(uuid.uuid5(CHUNK_NAMESPACE, chunk_id)) for chunk_id in chunks["chunk_id"]]


def self_retrieval_check(client: QdrantClient, name: str, ids: list[str], vectors: np.ndarray, sample_size: int = 5) -> None:
    """Zero-cost check (no OpenAI call): query with a stored point's own vector
    and confirm that point comes back as its own top match. This only proves
    the vectors were stored and are searchable — it says nothing about whether
    the payload indexes exist or work, since a query by vector doesn't use them."""
    rng = np.random.default_rng(0)
    sample = rng.choice(len(ids), size=sample_size, replace=False)
    for i in sample:
        results = client.query_points(collection_name=name, query=vectors[i].tolist(), limit=1).points
        top = results[0]
        ok = top.id == ids[i] and top.score > 0.999
        print(f"  [{'ok' if ok else 'MISMATCH'}] {ids[i]}: top hit score={top.score:.4f}")


def ann_recall_check(client: QdrantClient, name: str, ids: list[str], vectors: np.ndarray, sample_size: int = 20, k: int = 10) -> None:
    """Compares Qdrant's approximate top-k neighbors against an exact NumPy
    top-k (vectors are unit length, so a dot product is cosine similarity),
    for a handful of random stored vectors. This is what actually tests search
    quality, not just storage."""
    rng = np.random.default_rng(1)
    sample = rng.choice(len(ids), size=sample_size, replace=False)
    id_by_position = {pos: point_id for pos, point_id in enumerate(ids)}

    recalls = []
    for i in sample:
        qdrant_hits = client.query_points(collection_name=name, query=vectors[i].tolist(), limit=k).points
        qdrant_ids = {hit.id for hit in qdrant_hits}

        scores = vectors @ vectors[i]
        exact_top_positions = np.argpartition(-scores, k)[:k]
        exact_ids = {id_by_position[p] for p in exact_top_positions}

        recalls.append(len(qdrant_ids & exact_ids) / k)

    print(f"  ANN recall@{k} over {sample_size} random queries: {np.mean(recalls):.3f} (1.0 = exact)")


def print_index_status(client: QdrantClient, name: str) -> None:
    info = client.get_collection(name)
    print(f"  points_count={info.points_count}  indexed_vectors_count={info.indexed_vectors_count}")


def main(chunks_file: str, embeddings_file: str) -> None:
    chunks = pd.read_json(chunks_file, lines=True)
    books = pd.read_json(BOOKS_PATH, lines=True).set_index("book_id")
    vectors = np.load(embeddings_file)
    assert len(vectors) == len(chunks), f"{len(vectors)} vectors for {len(chunks)} chunks"

    name = collection_name(embeddings_file)
    client = get_client()

    create_collection(client, name, vectors.shape[1])
    create_payload_indexes(client, name)

    ids = build_ids(chunks)
    payloads = build_payloads(chunks, books)
    client.upload_collection(collection_name=name, vectors=vectors, payload=payloads, ids=ids, wait=True)

    print(f"stored {len(ids)} points in collection {name!r} at {QDRANT_URL}")
    print_index_status(client, name)

    print("self-retrieval check (proves storage, not indexing):")
    self_retrieval_check(client, name, ids, vectors)

    print("ANN recall check (proves search quality):")
    ann_recall_check(client, name, ids, vectors)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
