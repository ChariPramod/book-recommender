"""RAG step 3: turn every chunk into a vector with OpenAI's text-embedding-3-small.

Usage:
    python -m src.embed data/processed/chunks_sentence_230_prefix_ov1.jsonl

Output (next to the chunks file, in data/processed/):
    embeddings_sentence_230_prefix_ov1.npy    one row per chunk, same order as the chunks file
    embeddings_sentence_230_prefix_ov1.json   which model, dimensions and chunks file made the vectors
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from src.config import BOOKS_PATH, EMBEDDING_MODEL

PRICE_PER_MILLION_TOKENS = 0.02    # text-embedding-3-small, standard tier
MAX_INPUTS_PER_REQUEST = 200       # the API allows up to 2048; 200 keeps each request small
MAX_TOKENS_PER_REQUEST = 250_000   # the API caps tokens per request (300k); stay under it
MAX_RETRIES = 6

# errors worth retrying: they are about load or the network, not about our input
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def output_paths(chunks_path: Path) -> tuple[Path, Path]:
    """chunks_X.jsonl -> embeddings_X.npy and embeddings_X.json, in the same folder."""
    name = chunks_path.stem.replace("chunks_", "embeddings_", 1)
    return chunks_path.with_name(f"{name}.npy"), chunks_path.with_name(f"{name}.json")


def make_batches(chunks: pd.DataFrame) -> list[list[int]]:
    """Group row positions into batches, limited by both input count and total tokens.
    A count-only limit is not enough: with the 'whole' strategy, 200 long summaries
    could go past the API's token limit per request."""
    batches, current, current_tokens = [], [], 0
    for position, n_tokens in enumerate(chunks["n_tokens"]):
        full = len(current) >= MAX_INPUTS_PER_REQUEST or current_tokens + n_tokens > MAX_TOKENS_PER_REQUEST
        if current and full:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(position)
        current_tokens += n_tokens
    if current:
        batches.append(current)
    return batches


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """One API call, with exponential backoff on temporary errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            # the API returns one item per input, each tagged with its input position;
            # sort by that index so vectors line up with texts even if the order changed
            items = sorted(response.data, key=lambda item: item.index)
            return np.array([item.embedding for item in items], dtype=np.float32)
        except RETRYABLE_ERRORS as error:
            wait = 2 ** attempt
            print(f"  {type(error).__name__}, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
    raise RuntimeError(f"gave up after {MAX_RETRIES} attempts")


def test_call(client: OpenAI) -> int:
    """Embed one sentence to confirm the key and model work before spending real money."""
    vector = embed_texts(client, ["A detective investigates a murder in a small town."])[0]
    print(f"test call ok: model={EMBEDDING_MODEL}, dimensions={len(vector)}, length={np.linalg.norm(vector):.4f}")
    return len(vector)


def confirm_cost(chunks: pd.DataFrame) -> bool:
    total_tokens = int(chunks["n_tokens"].sum())
    cost = total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS
    print(f"about to embed {len(chunks):,} chunks, {total_tokens:,} tokens, estimated cost ${cost:.2f}")
    return input("proceed? type yes: ").strip().lower() == "yes"


def check_vectors(vectors: np.ndarray, chunks: pd.DataFrame) -> None:
    """Every chunk has exactly one vector, and OpenAI vectors come back with length 1.
    Length 1 means a dot product is the same as cosine similarity."""
    assert len(vectors) == len(chunks), f"{len(vectors)} vectors for {len(chunks)} chunks"
    lengths = np.linalg.norm(vectors, axis=1)
    print(f"vector lengths: min {lengths.min():.4f}, max {lengths.max():.4f} (expect about 1.0)")
    assert np.allclose(lengths, 1.0, atol=1e-3), "some vectors are not normalized"


def sanity_search(client: OpenAI, vectors: np.ndarray, chunks: pd.DataFrame) -> None:
    """Not an eval, just a look: do plausible books come back for plain queries?
    Cosine similarity by hand with NumPy, then each book scored by its best chunk."""
    books = pd.read_json(BOOKS_PATH, lines=True).set_index("book_id")
    queries = [
        "recommend some superpower fiction books",
        "a detective solves a poisoning murder",
        "a young wizard goes to a school of magic",
    ]
    query_vectors = embed_texts(client, queries)
    for query, query_vector in zip(queries, query_vectors):
        scores = vectors @ query_vector    # one dot product per chunk = cosine similarity
        best_per_book = (
            pd.DataFrame({"book_id": chunks["book_id"], "score": scores})
            .groupby("book_id")["score"].max()
            .sort_values(ascending=False)
            .head(5)
        )
        print(f"\nquery: {query}")
        for book_id, score in best_per_book.items():
            book = books.loc[book_id]
            print(f"  {score:.3f}  {book['title']} by {book['author']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a chunks file with OpenAI.")
    parser.add_argument("chunks_file", type=Path)
    args = parser.parse_args()

    load_dotenv()          # reads OPENAI_API_KEY from .env into the environment
    client = OpenAI()      # picks up OPENAI_API_KEY automatically

    chunks = pd.read_json(args.chunks_file, lines=True)
    npy_path, meta_path = output_paths(args.chunks_file)

    if npy_path.exists():
        # never pay twice: reuse the saved vectors, still run the checks
        print(f"{npy_path} already exists, skipping the API calls")
        vectors = np.load(npy_path)
    else:
        dimensions = test_call(client)
        if not confirm_cost(chunks):
            print("cancelled, nothing was embedded")
            return

        texts = chunks["embedded_text"].tolist()
        batches = make_batches(chunks)
        parts = []
        started = time.time()
        for number, batch in enumerate(batches, start=1):
            parts.append(embed_texts(client, [texts[i] for i in batch]))
            if number % 20 == 0 or number == len(batches):
                print(f"  batch {number}/{len(batches)} done, {time.time() - started:.0f}s")
        vectors = np.vstack(parts)

        np.save(npy_path, vectors)
        meta = {
            "model": EMBEDDING_MODEL,
            "dimensions": dimensions,
            "chunks_file": str(args.chunks_file),
            "count": len(vectors),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"saved {len(vectors):,} vectors to {npy_path}")

    check_vectors(vectors, chunks)
    sanity_search(client, vectors, chunks)


if __name__ == "__main__":
    main()