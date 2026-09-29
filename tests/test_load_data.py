import pandas as pd
import pytest

from src.load_data import clean_markup, deduplicate, drop_junk, load_raw, split_genres


@pytest.fixture(scope="module")
def cleaned_books() -> pd.DataFrame:
    df = load_raw()
    df["summary"] = df["summary"].apply(clean_markup)
    df = drop_junk(df)
    df = deduplicate(df)
    df = split_genres(df)
    return df


def test_real_books_survive_cleaning(cleaned_books):
    titles = set(cleaned_books["title"])
    for title in ["Area 7", "Century", "Dragon's Egg"]:
        assert title in titles


def test_junk_books_are_dropped(cleaned_books):
    titles = set(cleaned_books["title"])
    for title in ["Deathstalker", "Conviction", "Nothing to Lose"]:
        assert title not in titles


def test_enders_game_appears_once_with_full_summary(cleaned_books):
    rows = cleaned_books[cleaned_books["title"] == "Ender's Game"]
    assert len(rows) == 1
    assert len(rows.iloc[0]["summary"].split()) == 1129


def test_clean_markup_removes_comment_missing_open_bracket():
    text = "Kinsey hunts the killer. !-- internal note --> She finds a letter."
    result = clean_markup(text)
    assert "!--" not in result
    assert "-->" not in result
    assert "She finds a letter." in result


def test_clean_markup_removes_stray_close_marker():
    text = "The truth comes out -->. Survivors flee the island."
    result = clean_markup(text)
    assert "-->" not in result
    assert "Survivors flee the island." in result


def test_clean_markup_leaves_normal_text_unchanged():
    text = "A quiet town hides a terrible secret."
    assert clean_markup(text) == text
