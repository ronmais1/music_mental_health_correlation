"""
RQ2:
Is alignment between a participant's favorite genre and the genre they listen to most
associated with better mental health outcomes?

------------------------------------------------------------
Operational definitions:
1) "Most listened genre":
   For each participant, we look at all columns that start with "Frequency [".
   We convert frequency categories to an ordinal scale:
   Never=0, Rarely=1, Sometimes=2, Very frequently=3
   Then, the genre with the highest score (row-wise) is chosen via idxmax().

2) "Alignment":
   Alignment = True if Fav genre == Most_Listened_Genre, otherwise False.

3) "Mental_Health_Index":
   Mental_Health_Index = mean(Anxiety, Depression, Insomnia, OCD)
   (Each variable is on a 0–10 scale, so the mean remains on a 0–10 scale.)

------------------------------------------------------------
Hypotheses:
H0 (null): mean Mental_Health_Index is the same in aligned and not-aligned participants.
H1 (alt) : mean Mental_Health_Index differs between aligned and not-aligned participants.

------------------------------------------------------------
Statistical test:
Independent samples t-test (Aligned vs Not aligned), alpha = 0.05

Notes about interpretation:
- If p < 0.05 → reject H0 (significant difference)
- If p >= 0.05 → fail to reject H0 (no significant difference)
"""

from pathlib import Path
import logging

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from utilities import load_data, basic_cleaning
from consts import HEALTH_COLS, FREQ_MAPPING, FREQ_PREFIX

def get_logger() -> logging.Logger:
    """
    Create a logger (instead of many print statements).
    This keeps output clean and professional (and easier to debug).
    """
    logger = logging.getLogger("q2_alignment")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def encode_genre_frequencies(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    """
    Step 3: Encode genre listening frequency columns to ordinal numeric values.

    We find all columns that start with "Frequency [".
    Then map the text values (Never/Rarely/...) to integers (0..3).
    """
    genre_cols = [c for c in df.columns if c.startswith(FREQ_PREFIX)]
    if not genre_cols:
        raise ValueError("No genre frequency columns found (columns starting with 'Frequency [').")

    df = df.copy()
    for col in genre_cols:
        df[col] = df[col].map(FREQ_MAPPING)

    logger.info(f"Encoded {len(genre_cols)} genre frequency columns.")
    logger.info("Sample of encoded genre columns (head):")
    logger.info("\n" + str(df[genre_cols].head()))
    return df, genre_cols


def compute_most_listened_genre(df: pd.DataFrame, genre_cols: list[str], logger: logging.Logger) -> pd.DataFrame:
    """
    Step 4: Compute the most listened genre per participant.

    We use idxmax(axis=1) to select the frequency column with the highest value in each row.
    Then we clean the column label to keep only the genre name.
    """
    df = df.copy()
    df["Most_Listened_Genre"] = df[genre_cols].idxmax(axis=1)

    df["Most_Listened_Genre"] = (
        df["Most_Listened_Genre"]
        .str.replace(r"Frequency \[", "", regex=True)
        .str.replace("]", "", regex=False)
    )

    logger.info("Fav genre vs Most_Listened_Genre (head):")
    logger.info("\n" + str(df[["Fav genre", "Most_Listened_Genre"]].head()))
    return df


def compute_alignment(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Step 5: Create Alignment boolean variable.
    Alignment is True if favorite genre equals most listened genre.
    """
    df = df.copy()
    df["Alignment"] = df["Fav genre"] == df["Most_Listened_Genre"]

    logger.info("Alignment sample (head):")
    logger.info("\n" + str(df[["Fav genre", "Most_Listened_Genre", "Alignment"]].head()))
    logger.info("Alignment counts:")
    logger.info("\n" + str(df["Alignment"].value_counts()))
    return df


def compute_mental_health_index(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Step 6: Create Mental_Health_Index.
    We use the mean of Anxiety, Depression, Insomnia, and OCD for each participant.
    """
    df = df.copy()
    df["Mental_Health_Index"] = df[HEALTH_COLS].mean(axis=1)

    logger.info("Mental health columns + index (head):")
    logger.info("\n" + str(df[HEALTH_COLS + ["Mental_Health_Index"]].head()))
    return df


def run_ttest(df: pd.DataFrame, logger: logging.Logger) -> tuple[float, float]:
    """
    Step 8: Statistical test.
    Independent samples t-test comparing Mental_Health_Index for:
    - aligned participants
    - not aligned participants
    """
    aligned = df[df["Alignment"] == True]["Mental_Health_Index"]
    not_aligned = df[df["Alignment"] == False]["Mental_Health_Index"]

    t_stat, p_value = ttest_ind(aligned, not_aligned, nan_policy="omit")

    logger.info("T-test results (Aligned vs Not aligned):")
    logger.info(f"t-statistic = {t_stat:.3f}")
    logger.info(f"p-value     = {p_value:.4f}")

    if p_value < 0.05:
        logger.info("Conclusion: Significant difference (p < 0.05). Reject H0.")
    else:
        logger.info("Conclusion: Not significant (p >= 0.05). Fail to reject H0.")

    return t_stat, p_value


def plot_boxplot(df: pd.DataFrame, out_path: Path, logger: logging.Logger, show: bool = False) -> None:
    """
    Step 7: Visualization (boxplot).
    We save the plot to a file so the script never "gets stuck" only showing a window.
    show=False prevents blocking/KeyboardInterrupt.
    """
    plt.figure(figsize=(6, 4))
    df.boxplot(column="Mental_Health_Index", by="Alignment")

    plt.title("Mental Health Index by Music Alignment")
    plt.suptitle("")
    plt.xlabel("Alignment (Favorite vs. Most Listened Genre)")
    plt.ylabel("Mental Health Index (0–10)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    logger.info(f"Saved plot to: {out_path}")

    if show:
        plt.show()

    plt.close()


def run_question_two() -> None:
    """
    Main pipeline (minimal): load → clean → encode → compute variables → test → plot → interpret.
    """
    logger = get_logger()

    # IMPORTANT:
    # The CSV must be located in the SAME folder as this script
    # because we build the path relative to this file.
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "mxmh_survey_results.csv"
    plot_path = script_dir / "alignment_boxplot.png"

    df = load_data(csv_path, logger)
    df = basic_cleaning(df, logger)
    df, genre_cols = encode_genre_frequencies(df, logger)
    df = compute_most_listened_genre(df, genre_cols, logger)
    df = compute_alignment(df, logger)
    df = compute_mental_health_index(df, logger)
    t_stat, p_value = run_ttest(df, logger)
    plot_boxplot(df, plot_path, logger, show=False)

    # -----------------------------
    # Interpretation (for submission)
    # -----------------------------
    logger.info("Interpretation:")
    if p_value < 0.05:
        logger.info(
            "There is a statistically significant difference in Mental_Health_Index "
            "between aligned and not-aligned participants in this sample."
        )
    else:
        logger.info(
            "There is no statistically significant difference in Mental_Health_Index "
            "between aligned and not-aligned participants in this sample."
        )
        
        
    logger.info("Done.")