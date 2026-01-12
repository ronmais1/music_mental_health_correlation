from pathlib import Path
import pandas as pd

from cod2 import (
    basic_cleaning,
    encode_genre_frequencies,
    compute_most_listened_genre,
    compute_alignment,
    compute_mental_health_index,
    get_logger,
)


def test_basic_cleaning_drops_missing_required():
    logger = get_logger()
    df = pd.DataFrame({
        "Age": [20, None],
        "Hours per day": [2, 3],
        "Anxiety": [5, 6],
        "Depression": [5, 6],
        "Insomnia": [5, 6],
        "OCD": [5, 6],
        "Fav genre": ["Rock", "Pop"],
        "Frequency [Rock]": ["Never", "Sometimes"],
    })
    cleaned = basic_cleaning(df, logger)
    assert cleaned.shape[0] == 1


def test_encode_genre_frequencies_creates_numeric():
    logger = get_logger()
    df = pd.DataFrame({
        "Age": [20],
        "Hours per day": [2],
        "Anxiety": [5],
        "Depression": [5],
        "Insomnia": [5],
        "OCD": [5],
        "Fav genre": ["Rock"],
        "Frequency [Rock]": ["Very frequently"],
    })
    df2, genre_cols = encode_genre_frequencies(df, logger)
    assert genre_cols == ["Frequency [Rock]"]
    assert df2["Frequency [Rock]"].iloc[0] == 3


def test_alignment_and_index_columns_exist():
    logger = get_logger()
    df = pd.DataFrame({
        "Age": [20],
        "Hours per day": [2],
        "Anxiety": [4],
        "Depression": [6],
        "Insomnia": [2],
        "OCD": [8],
        "Fav genre": ["Rock"],
        "Frequency [Rock]": [3],
        "Frequency [Pop]": [1],
    })
    genre_cols = ["Frequency [Rock]", "Frequency [Pop]"]
    df = compute_most_listened_genre(df, genre_cols, logger)
    df = compute_alignment(df, logger)
    df = compute_mental_health_index(df, logger)

    assert "Alignment" in df.columns
    assert "Mental_Health_Index" in df.columns
    assert abs(df["Mental_Health_Index"].iloc[0] - 5.0) < 1e-9
