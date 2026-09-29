import json

import pytest

from src.chunking import chunk_book, count_tokens, make_prefix, pack_sentences, split_sentences
from src.config import BOOKS_PATH, CHUNK_MAX_TOKENS, MIN_CHUNK_TOKENS, OVERLAP_SENTENCES


@pytest.fixture(scope="module")
def real_books() -> list[dict]:
    return [json.loads(line) for line in open(BOOKS_PATH)]


@pytest.fixture(scope="module")
def real_chunks(real_books) -> list[dict]:
    # chunk_book (unlike chunk_books) leaves the internal "_kind" tag in place,
    # which these tests use to know which chunks are allowed to be exceptions
    records = []
    for book in real_books:
        book_records, _ = chunk_book(book, "sentence", CHUNK_MAX_TOKENS)
        records.extend(book_records)
    return records


def test_raw_text_appears_in_own_book_summary(real_books, real_chunks):
    books_by_id = {b["book_id"]: b for b in real_books}
    for chunk in real_chunks:
        assert chunk["raw_text"] in books_by_id[chunk["book_id"]]["summary"]


def test_no_chunk_exceeds_budget_except_oversized_and_merged(real_chunks):
    unexplained = [
        c for c in real_chunks if c["n_tokens"] > CHUNK_MAX_TOKENS and c["_kind"] == "normal"
    ]
    assert unexplained == []


def test_consecutive_chunks_share_overlap_except_known_exceptions(real_books):
    for book in real_books:
        prefix = make_prefix(book["title"], book["author"], book["genres"])
        sentences = split_sentences(book["summary"])
        groups, oversized_flags = pack_sentences(sentences, CHUNK_MAX_TOKENS, prefix, OVERLAP_SENTENCES)

        for i in range(len(groups) - 1):
            # exception 1: no overlap is carried into or out of an oversized-sentence chunk
            if oversized_flags[i] or oversized_flags[i + 1]:
                continue

            expected_overlap = groups[i][-OVERLAP_SENTENCES:]
            actual_start = groups[i + 1][:OVERLAP_SENTENCES]
            if actual_start == expected_overlap:
                continue

            # exception 2: the overlap was dropped because it didn't fit with
            # the next sentence — confirm that's really why, not a real bug
            combined = expected_overlap + groups[i + 1][:1]
            assert count_tokens(prefix + " ".join(combined)) > CHUNK_MAX_TOKENS, (
                f"unexplained overlap mismatch in {book['book_id']} at chunk {i}"
            )


def test_no_cross_book_contamination():
    book_a = {
        "book_id": "bAAAA",
        "title": "Alpha",
        "author": "Author A",
        "genres": ["Fiction"],
        "summary": "The alpha hero saves the alpha village. The alpha village celebrates.",
    }
    book_b = {
        "book_id": "bBBBB",
        "title": "Beta",
        "author": "Author B",
        "genres": ["Fiction"],
        "summary": "The beta hero saves the beta village. The beta village celebrates.",
    }
    records_a, _ = chunk_book(book_a, "sentence", CHUNK_MAX_TOKENS)
    records_b, _ = chunk_book(book_b, "sentence", CHUNK_MAX_TOKENS)

    for record in records_a:
        assert "beta" not in record["raw_text"].lower()
    for record in records_b:
        assert "alpha" not in record["raw_text"].lower()


def test_short_summary_fits_in_exactly_one_chunk():
    book = {
        "book_id": "b00001",
        "title": "Short Book",
        "author": "Some Author",
        "genres": ["Fiction"],
        "summary": "A short summary that easily fits the budget.",
    }
    records, stats = chunk_book(book, "sentence", CHUNK_MAX_TOKENS)
    assert len(records) == 1
    assert stats["oversized_sentences"] == 0


def test_oversized_sentence_becomes_one_unsplit_chunk():
    long_sentence = "This is a very long sentence. " * 40
    groups, oversized_flags = pack_sentences([long_sentence.strip()], max_tokens=20)
    assert len(groups) == 1
    assert groups[0] == [long_sentence.strip()]
    assert oversized_flags == [True]


def test_use_prefix_false_makes_embedded_text_equal_raw_text():
    book = {
        "book_id": "b00001",
        "title": "Some Title",
        "author": "Some Author",
        "genres": ["Fiction", "Adventure"],
        "summary": "One sentence here. Another sentence follows. And a third one too.",
    }
    records, _ = chunk_book(book, "sentence", CHUNK_MAX_TOKENS, use_prefix=False)
    for record in records:
        assert record["embedded_text"] == record["raw_text"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "He works for the U.N. in New York. Dr. Smith agrees. The U.S. sent aid.",
            ["He works for the U.N. in New York.", "Dr. Smith agrees.", "The U.S. sent aid."],
        ),
        (
            "J. R. R. Tolkien wrote this. It was published in 1954.",
            ["J. R. R. Tolkien wrote this.", "It was published in 1954."],
        ),
        (
            "St. Mary's Church burned down. Everyone was shocked.",
            ["St. Mary's Church burned down.", "Everyone was shocked."],
        ),
        (
            "Lt. Dan saved the platoon. He earned a medal.",
            ["Lt. Dan saved the platoon.", "He earned a medal."],
        ),
    ],
)
def test_split_sentences_ignores_abbreviations_and_initials(text, expected):
    assert split_sentences(text) == expected
