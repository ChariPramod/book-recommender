"""Central settings for the book recommender pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "cmu_books_clean.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BOOKS_PATH = PROCESSED_DIR / "books.jsonl"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"  # small, cheap OpenAI chat model — check this is still current
ENCODING = "cl100k_base"

MODEL_MAX_TOKENS = 8191
CHUNK_MAX_TOKENS = 230  # a quality choice we will tune later, not a model limit
OVERLAP_SENTENCES = 1  # experiment: 0 vs 1
MIN_CHUNK_TOKENS = 80
USE_PREFIX = True  # experiment: True vs False
STRATEGY = "sentence"  # experiment: "sentence" vs "whole"

EMBEDDING_COST_PER_1M_TOKENS = 0.02

# Qdrant Cloud free cluster; credentials live in .env, never in this file or printed anywhere
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "sentence_230_prefix_ov1"

# retrieval + reranking
RETRIEVE_LIMIT = 30  # books pulled from Qdrant (one best chunk each) before reranking
RERANK_MODEL = "rerank-v4.0-fast"  # Cohere's low-latency rerank model
RERANK_TOP_N = 10  # books kept after reranking, before the threshold
RERANK_SUMMARY_CHARS = 2000  # summary trimmed to this before being sent to the reranker
MATCH_THRESHOLD = 0.3  # minimum rerank score to show a book — NOT YET TUNED
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


def require_qdrant_credentials() -> None:
    # a clear failure at the point Qdrant is actually needed, not at import
    # time — importing config.py (which every module does, including tests
    # that have nothing to do with Qdrant) must not require these to be set
    missing = [
        name
        for name, value in [("QDRANT_URL", QDRANT_URL), ("QDRANT_API_KEY", QDRANT_API_KEY)]
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing from .env: {', '.join(missing)}. Add them before running src.store.")


def collection_name(embeddings_file: Path) -> str:
    # one collection per experiment config, e.g. "sentence_230_prefix_ov1", so
    # storing one experiment's vectors never touches another's collection
    return Path(embeddings_file).stem.replace("embeddings_", "", 1)


def chunks_path(strategy: str, max_tokens: int, use_prefix: bool, overlap_sentences: int) -> Path:
    prefix_tag = "prefix" if use_prefix else "noprefix"
    return PROCESSED_DIR / f"chunks_{strategy}_{max_tokens}_{prefix_tag}_ov{overlap_sentences}.jsonl"


def embeddings_path(chunks_file: Path) -> Path:
    # embeddings share the chunks file's exact naming, just swapped prefix/extension,
    # so it's obvious at a glance which chunks a given embeddings file came from
    stem = Path(chunks_file).stem.replace("chunks_", "embeddings_", 1)
    return PROCESSED_DIR / f"{stem}.npy"
