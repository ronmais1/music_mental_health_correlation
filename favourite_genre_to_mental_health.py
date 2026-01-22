from pathlib import Path
import logging
import pandas as pd
from scipy.stats import ttest_ind
from utilities import load_data, basic_cleaning, get_logger
from visualize import plot_boxplot, plot_alignment_means, plot_disorders_by_alignment
from consts import HEALTH_COLS, FREQ_MAPPING, FREQ_PREFIX


def encode_genre_frequencies(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode genre listening frequency columns to ordinal numeric values.

    We find all columns that start with "Frequency [".
    Then map the text values (Never/Rarely/...) to integers (0..3).
    """
    genre_cols = [c for c in df.columns if c.startswith(FREQ_PREFIX)]
    if not genre_cols:
        raise ValueError("No genre frequency columns found (columns starting with 'Frequency [').")

    df = df.copy()
    for col in genre_cols:
        df[col] = df[col].astype(str).str.strip().map(FREQ_MAPPING)
        
    # Check if mapping created NaN values (unmapped labels)
    missing = df[genre_cols].isna().sum().sum()
    if missing > 0:
        logger.warning(
        f"Encoding warning: {missing} unmapped frequency values became NaN. "
        "Check FREQ_MAPPING or raw data labels."
    )


    logger.info(f"Encoded {len(genre_cols)} genre frequency columns.")
    logger.info("Sample of encoded genre columns (head):")
    logger.info("\n" + str(df[genre_cols].head()))
    return df, genre_cols


def compute_most_listened_genre(df: pd.DataFrame, genre_cols: list[str], logger: logging.Logger) -> pd.DataFrame:
    """
    Compute the most listened genre per participant.

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
    Create Alignment boolean variable.
    Alignment is True if favorite genre equals most listened genre.
    """
    df = df.copy()
    df["Alignment"] = df["Fav genre"] == df["Most_Listened_Genre"]

    logger.info("Alignment sample (head):")
    logger.info("\n" + str(df[["Fav genre", "Most_Listened_Genre", "Alignment"]].head()))
    logger.info("Alignment counts:")
    logger.info("\n" + str(df["Alignment"].value_counts()))
    return df

def summarize_alignment_distribution(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Q2 - Descriptive step:
    Summarize how common alignment is in the sample.

    We log:
    - counts of Alignment (True/False)
    - percentages of Alignment (True/False)
    """
    counts = df["Alignment"].value_counts(dropna=False)
    percents = df["Alignment"].value_counts(normalize=True, dropna=False) * 100

    logger.info("=== Q2: Alignment Distribution ===")
    logger.info("Counts (N):")
    logger.info("\n" + str(counts))

    logger.info("Percentages (%):")
    logger.info("\n" + str(percents.round(2)))



def compute_mental_health_index(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Create Mental_Health_Index.
    We use the mean of Anxiety, Depression, Insomnia, and OCD for each participant.
    """
    df = df.copy()
    df["Mental_Health_Index"] = df[HEALTH_COLS].mean(axis=1)

    logger.info("Mental health columns + index (head):")
    logger.info("\n" + str(df[HEALTH_COLS + ["Mental_Health_Index"]].head()))
    return df

def summarize_alignment_statistic(df, logger):
    """
    Descriptive statistics of Mental Health Index by Alignment.
    """
    summary = (
        df.groupby("Alignment")["Mental_Health_Index"]
        .agg(["count", "mean", "std"])
    )

    logger.info("=== Q2: Mental Health Index by Alignment ===")
    logger.info(f"\n{summary}")

    return summary


def run_ttest(df: pd.DataFrame, logger: logging.Logger) -> tuple[float, float]:
    """
    Statistical test.
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

def run_ttests_per_disorder(df: pd.DataFrame, logger: logging.Logger):
    """
    Run independent t-tests for each mental health variable
    comparing aligned vs. not-aligned participants.
    """
    logger.info("=== Q2: T-tests per Mental Health Variable ===")

    for col in HEALTH_COLS:
        aligned = df[df["Alignment"] == True][col]
        not_aligned = df[df["Alignment"] == False][col]

        t_stat, p_value = ttest_ind(aligned, not_aligned, nan_policy="omit")

        logger.info(f"\n{col}:")
        logger.info(f"  t = {t_stat:.3f}")
        logger.info(f"  p = {p_value:.4f}")

        if p_value < 0.05:
            logger.info("  Result: Significant difference (p < 0.05)")
        else:
            logger.info("  Result: Not significant (p ≥ 0.05)")




def run_question_two(logger) -> None:
    """
    Main pipeline (minimal): load the data → clean the data → encode the data → compute variables (genre, alignment, mental index) → test → plot (visualize) → interpret.
    """
    # IMPORTANT:
    # The CSV must be located in the SAME folder as this script
    # because we build the path relative to this file.
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "mxmh_survey_results.csv"

    df = load_data(csv_path, logger)
    df = basic_cleaning(df, logger, HEALTH_COLS)
    df, genre_cols = encode_genre_frequencies(df, logger)
    df = compute_most_listened_genre(df, genre_cols, logger)
    
    df = compute_alignment(df, logger)
    summarize_alignment_distribution(df, logger)
    
    df = compute_mental_health_index(df, logger)
    summarize_alignment_statistic(df, logger)

    t_stat, p_value = run_ttest(df, logger)
    run_ttests_per_disorder(df, logger)
    
    plot_alignment_means(df, logger)
    plot_boxplot(df, logger)
    plot_disorders_by_alignment(df, HEALTH_COLS, logger)
    plot_alignment_means(df, logger)
    


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
  

 
