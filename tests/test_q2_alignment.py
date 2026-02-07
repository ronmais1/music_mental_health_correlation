import pandas as pd
import pytest

from utilities import get_logger
from favourite_genre_to_mental_health import (
    compute_most_listened_genre,
    compute_alignment,
    compute_mental_health_index,
)
from consts import ALIGNMENT, MENTAL_HEALTH_INDEX


@pytest.fixture
def logger():
    """
    Provide a project logger instance for tests.
    """
    return get_logger()


def test_q2_alignment_and_mental_health_index_pipeline(logger):
    """
    Research Question 2 - core pipeline test (unit-level):

    We validate that the pipeline correctly:
    1) Identifies the most listened genre using the max frequency column.
    2) Computes Alignment = (Fav genre == Most_Listened_Genre).
    3) Computes MENTAL_HEALTH_INDEX as the mean of the 4 health columns.

    The dataset here is synthetic and minimal so expected values are deterministic.
    """

    # -------------------------
    # Arrange: minimal dataset
    # -------------------------
    df = pd.DataFrame(
        {
            "Age": [20],
            "Hours per day": [2],
            "Anxiety": [4],
            "Depression": [6],
            "Insomnia": [2],
            "OCD": [8],
            "Fav genre": ["Rock"],
            # Frequencies are already numeric here because this test focuses on the Q2 pipeline steps
            "Frequency [Rock]": [3],
            "Frequency [Pop]": [1],
        }
    )

    genre_cols = ["Frequency [Rock]", "Frequency [Pop]"]

    # ---------------
    # Act: run steps
    # ---------------
    df = compute_most_listened_genre(df, genre_cols, logger)
    df = compute_alignment(df, logger)
    df = compute_mental_health_index(df, logger)

    # ----------------------------
    # Assert: columns were created
    # ----------------------------
    assert "Most_Listened_Genre" in df.columns
    assert ALIGNMENT in df.columns
    assert MENTAL_HEALTH_INDEX in df.columns

    # --------------------------------
    # Assert: computed values are right
    # --------------------------------
    assert df.loc[0, "Most_Listened_Genre"] == "Rock"

    # IMPORTANT:
    # Alignment value is often numpy.bool_ (np.True_/np.False_), so we must not use "is True".
    assert bool(df.loc[0, ALIGNMENT]) is True

    # Mean of (4 + 6 + 2 + 8) / 4 = 5.0
    assert df.loc[0, MENTAL_HEALTH_INDEX] == pytest.approx(5.0, abs=1e-12)
