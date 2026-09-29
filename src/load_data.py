"""RAG step 1: load the raw CSV, clean it, and save books.jsonl."""

import os
import re

import pandas as pd

from src.config import BOOKS_PATH, PROCESSED_DIR, RAW_CSV_PATH


def load_raw(path=RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # book_id comes from the original row order, before any filtering, so it
    # never shifts when a cleaning rule changes later (vectors, Qdrant points,
    # and the eval set all reference book_id).
    df["book_id"] = [f"b{i:05d}" for i in range(1, len(df) + 1)]
    return df


def clean_markup(text: str) -> str:
    text = text.split("<!--", 1)[0]
    text = re.sub(r"!--.*?-->", " ", text)  # comments that lost their "<"
    text = re.sub(r"<!-+|-->|~Plot outline description~", " ", text)  # leftover fragments
    text = re.sub(r"={2,}", " ", text)
    text = text.replace("|----", " ").replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def drop_junk(df: pd.DataFrame) -> pd.DataFrame:
    word_count = df["summary"].str.split().str.len()
    is_short = word_count < 10
    is_placeholder = df["summary"] == "To be added."
    before = len(df)
    df = df[~(is_short | is_placeholder)]
    removed = before - len(df)
    print(f"drop_junk: removed {removed} rows ({is_placeholder.sum()} of them placeholders)")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # same (title, author): keep the row with the longest summary
    word_count = df["summary"].str.split().str.len()
    df = df.assign(_word_count=word_count).sort_values("_word_count", ascending=False)
    df = df.drop_duplicates(["title", "author"], keep="first").drop(columns="_word_count")
    after_title_author = len(df)

    # identical summary on different books: at least one is wrong, drop all copies
    df = df[~df["summary"].duplicated(keep=False)]
    after_summary = len(df)

    print(
        f"deduplicate: removed {before - after_title_author} duplicate (title, author) rows, "
        f"{after_title_author - after_summary} rows with a summary shared by a different book"
    )
    return df.sort_index()


def split_genres(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["genres"] = df["genres"].str.split(" | ", regex=False)
    return df


def save_books(df: pd.DataFrame, path=BOOKS_PATH) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def main() -> None:
    df = load_raw()
    print(f"loaded {len(df)} rows")

    cleaned = df["summary"].apply(clean_markup)
    changed = df["summary"] != cleaned
    print(f"clean_markup: changed {changed.sum()} summaries")
    window = 40
    for before, after in zip(df.loc[changed, "summary"].head(3), cleaned.loc[changed].head(3)):
        # print the text around the first difference, not just the first 100
        # characters, since markup is often further into the summary
        pos = len(os.path.commonprefix([before, after]))
        start = max(pos - window, 0)
        print(f"before: ...{before[start:pos + window]}...")
        print(f"after:  ...{after[start:pos + window]}...\n")
    df["summary"] = cleaned

    df = drop_junk(df)
    df = deduplicate(df)
    df = split_genres(df)

    save_books(df)
    print(f"saved {len(df)} books to {BOOKS_PATH}")


if __name__ == "__main__":
    main()
