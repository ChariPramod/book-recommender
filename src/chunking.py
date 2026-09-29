"""RAG step 2: turn each book's summary into one or more embeddable chunks."""

import re

import pandas as pd
import tiktoken

from src.config import (
    BOOKS_PATH,
    CHUNK_MAX_TOKENS,
    ENCODING,
    MIN_CHUNK_TOKENS,
    MODEL_MAX_TOKENS,
    OVERLAP_SENTENCES,
    PROCESSED_DIR,
    STRATEGY,
    USE_PREFIX,
    chunks_path,
)

_ENCODER = tiktoken.get_encoding(ENCODING)

_ABBREVIATIONS = {
    "U.N.", "U.S.", "Dr.", "Mr.", "Mrs.", "St.", "Lt.", "Jr.", "Sr.", "Mt.",
    "Capt.", "Col.", "Gen.", "Sgt.", "Prof.", "Rev.", "Ms.", "vs.",
}
# a sentence boundary candidate: ./!/? followed by whitespace and a capital
# letter or an opening quote (a lowercase letter after the space means the
# punctuation wasn't really ending a sentence, e.g. an abbreviation or ellipsis)
_BOUNDARY = re.compile(r"[.!?](?=\s+[A-Z\"'‘“])")
_TRAILING_WORD = re.compile(r"(\S+)$")


def split_sentences(text: str) -> list[str]:
    sentences = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        word_match = _TRAILING_WORD.search(text[:end])
        preceding_word = word_match.group(1) if word_match else ""
        is_abbreviation = preceding_word in _ABBREVIATIONS
        # a bare initial like "J." — a single capital letter plus a period —
        # doesn't end a sentence either, e.g. "J. R. R. Tolkien"
        is_initial = len(preceding_word) == 2 and preceding_word[0].isupper() and preceding_word[1] == "."
        if is_abbreviation or is_initial:
            continue
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def make_prefix(title: str, author: str, genres: list[str]) -> str:
    return f"{title} by {author} ({', '.join(genres)}): "


def pack_sentences(
    sentences: list[str],
    max_tokens: int,
    prefix: str = "",
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> tuple[list[list[str]], list[bool]]:
    """Greedily packs sentences into chunks so that `prefix + " ".join(chunk)`
    never exceeds `max_tokens`, with `overlap_sentences` of overlap between
    consecutive chunks. Every "would this fit" check counts the tokens of the
    exact text that will be saved, not a sum of per-sentence token counts —
    those can differ by a token or two at a join boundary, since tokenization
    isn't strictly additive. Returns (chunks, oversized_flags), where
    oversized_flags[i] is True if chunks[i] is a single sentence that alone
    (with the prefix) exceeded the budget."""

    def tokens_for(group: list[str]) -> int:
        return count_tokens(prefix + " ".join(group))

    chunks: list[list[str]] = []
    oversized_flags: list[bool] = []
    current: list[str] = []

    for sentence in sentences:
        if tokens_for([sentence]) > max_tokens:
            # this sentence alone (with the prefix) already exceeds the
            # budget: leave it unsplit, as its own chunk (safe, since
            # MODEL_MAX_TOKENS is far larger)
            if current:
                chunks.append(current)
                oversized_flags.append(False)
                current = []
            chunks.append([sentence])
            oversized_flags.append(True)
            continue

        if current and tokens_for(current + [sentence]) > max_tokens:
            chunks.append(current)
            oversized_flags.append(False)
            # overlap: the next chunk starts with the previous chunk's last
            # `overlap_sentences` sentences, unless the overlap alone already
            # leaves no room for this one
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            if overlap and tokens_for(overlap + [sentence]) > max_tokens:
                current = []
            else:
                current = list(overlap)

        current.append(sentence)

    if current:
        chunks.append(current)
        oversized_flags.append(False)

    return chunks, oversized_flags


def _merge_small_tail(
    groups: list[list[str]], prefix: str = "", overlap_sentences: int = OVERLAP_SENTENCES
) -> tuple[list[list[str]], bool]:
    if len(groups) < 2:
        return groups, False

    tail_tokens = count_tokens(" ".join(groups[-1]))
    if tail_tokens >= MIN_CHUNK_TOKENS:
        return groups, False

    # drop the overlap sentences shared with the previous group before
    # merging, so they aren't duplicated in the merged text
    n = overlap_sentences
    has_overlap = n > 0 and len(groups[-1]) > n and groups[-1][:n] == groups[-2][-n:]
    new_material = groups[-1][n:] if has_overlap else groups[-1]
    merged = groups[-2] + new_material

    if count_tokens(prefix + " ".join(merged)) > MODEL_MAX_TOKENS:
        return groups, False

    return groups[:-2] + [merged], True


def _truncate_to_max_tokens(text: str, max_tokens: int) -> str:
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens])


def chunk_book(
    book: dict,
    strategy: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    use_prefix: bool = USE_PREFIX,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> tuple[list[dict], dict]:
    # always compute the real prefix and pack against it, even when
    # use_prefix is False — USE_PREFIX must only change embedded_text, never
    # where chunk boundaries fall, so the prefix on/off experiment isn't
    # secretly also an experiment on chunk size
    prefix = make_prefix(book["title"], book["author"], book["genres"])
    applied_prefix = prefix if use_prefix else ""
    stats = {"oversized_sentences": 0, "merged_tail": False, "truncated_title": None}

    if strategy == "whole":
        raw_text = book["summary"]
        embedded_text = applied_prefix + raw_text
        n_tokens = count_tokens(embedded_text)
        kind = "normal"
        if n_tokens > MODEL_MAX_TOKENS:
            embedded_text = _truncate_to_max_tokens(embedded_text, MODEL_MAX_TOKENS)
            n_tokens = count_tokens(embedded_text)
            stats["truncated_title"] = book["title"]
            kind = "truncated"
        return (
            [
                {
                    "chunk_id": f"{book['book_id']}-0",
                    "book_id": book["book_id"],
                    "chunk_index": 0,
                    "raw_text": raw_text,
                    "embedded_text": embedded_text,
                    "n_tokens": n_tokens,
                    "_kind": kind,
                }
            ],
            stats,
        )

    if strategy == "sentence":
        sentences = split_sentences(book["summary"])
        groups, oversized_flags = pack_sentences(sentences, max_tokens, prefix, overlap_sentences)
        stats["oversized_sentences"] = sum(oversized_flags)
        groups, stats["merged_tail"] = _merge_small_tail(groups, prefix, overlap_sentences)
        if stats["merged_tail"]:
            # a merge always combines two non-oversized groups (an oversized
            # group's tail is always well above MIN_CHUNK_TOKENS, so it's
            # never merge-eligible) into one normal one
            oversized_flags = oversized_flags[:-2] + [False]
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    last_index = len(groups) - 1
    records = []
    for index, group in enumerate(groups):
        raw_text = " ".join(group)
        embedded_text = applied_prefix + raw_text
        if index == last_index and stats["merged_tail"]:
            kind = "merged"
        elif oversized_flags[index]:
            kind = "oversized"
        else:
            kind = "normal"
        records.append(
            {
                "chunk_id": f"{book['book_id']}-{index}",
                "book_id": book["book_id"],
                "chunk_index": index,
                "raw_text": raw_text,
                "embedded_text": embedded_text,
                "n_tokens": count_tokens(embedded_text),
                "_kind": kind,
            }
        )
    return records, stats


def chunk_books(
    strategy: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    use_prefix: bool = USE_PREFIX,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list[dict]:
    books = pd.read_json(BOOKS_PATH, lines=True).to_dict(orient="records")

    all_records: list[dict] = []
    total_oversized = 0
    total_merged = 0
    truncated_titles: list[str] = []
    for book in books:
        records, stats = chunk_book(book, strategy, max_tokens, use_prefix, overlap_sentences)
        all_records.extend(records)
        total_oversized += stats["oversized_sentences"]
        total_merged += 1 if stats["merged_tail"] else 0
        if stats["truncated_title"]:
            truncated_titles.append(stats["truncated_title"])

    over_budget = [r for r in all_records if r["n_tokens"] > max_tokens]
    over_budget_by_kind: dict[str, int] = {}
    for record in over_budget:
        over_budget_by_kind[record["_kind"]] = over_budget_by_kind.get(record["_kind"], 0) + 1
    other = over_budget_by_kind.get("normal", 0)

    print(
        f"chunked {len(books)} books into {len(all_records)} chunks "
        f"(strategy={strategy}, max_tokens={max_tokens})"
    )
    print(f"{total_oversized} sentences alone exceeded the budget and became their own chunk")
    print(f"{total_merged} books had a small final chunk merged into the previous one")
    print(
        f"{len(over_budget)} chunks exceed {max_tokens} tokens: "
        f"{over_budget_by_kind.get('oversized', 0)} oversized sentences, "
        f"{over_budget_by_kind.get('merged', 0)} merged tails, "
        f"{other} other (should be 0)"
    )
    if truncated_titles:
        print(f"{len(truncated_titles)} whole-summary chunks exceeded MODEL_MAX_TOKENS and were truncated:")
        for title in truncated_titles:
            print(f"  - {title}")

    for record in all_records:
        del record["_kind"]
    return all_records


def save_chunks(records: list[dict], path) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_json(path, orient="records", lines=True, force_ascii=False)


def main(
    strategy: str = STRATEGY,
    max_tokens: int = CHUNK_MAX_TOKENS,
    use_prefix: bool = USE_PREFIX,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> None:
    records = chunk_books(strategy, max_tokens, use_prefix, overlap_sentences)
    path = chunks_path(strategy, max_tokens, use_prefix, overlap_sentences)
    save_chunks(records, path)
    print(f"saved {len(records)} chunks to {path}")


if __name__ == "__main__":
    main()
