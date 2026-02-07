from pathlib import Path
import logging
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from utilities import calculate_distress_index, load_data, basic_cleaning, split_by_alignment
from visualize import plot_boxplot, plot_alignment_means, plot_disorders_by_alignment
from consts import (
    AGGREGATED_HEALTH_COLS,
    HEALTH_COLS,
    FREQ_MAPPING,
    FREQ_PREFIX,
    ALIGNMENT,
    MENTAL_HEALTH_INDEX,
)


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
    df = df.copy()

    def pick_best_genre(row):
        # Find the max value in the row
        max_val = row[genre_cols].max()
        
        # Get all genres that share that max value
        ties = row[genre_cols][row[genre_cols] == max_val].index.tolist()
        without_freq_ties = [g.replace("Frequency [", "").replace("]", "") for g in ties]
        
        if len(without_freq_ties) == 1:
            return without_freq_ties[0]
        
        # If there's a tie, check if 'Fav genre' is among the winners
        if row["Fav genre"] in without_freq_ties:
            return row["Fav genre"]
        
        # Otherwise, just pick the first one from the tie list
        return without_freq_ties[0]

    def get_num_of_most_listened_genres(row):
        # Find the max value in the row
        max_val = row[genre_cols].max()
        
        # Get all genres that share that max value
        ties = row[genre_cols][row[genre_cols] == max_val].index.tolist()

        return len(ties)

    # Apply the logic row by row
    # We include "Fav genre" in the axis=1 apply so the function can see it
    df["Most_Listened_Genre"] = df[genre_cols + ["Fav genre"]].apply(pick_best_genre, axis=1)
    df["Number_Of_Most_Listened_Genre"] = df[genre_cols].apply(get_num_of_most_listened_genres, axis=1)

    logger.info("Fav genre vs Most_Listened_Genre (head):")
    logger.info("\n" + str(df[["Fav genre", "Most_Listened_Genre"]].head()))
    
    return df

def compute_alignment(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Create Alignment boolean variable.
    Alignment is True if favorite genre equals most listened genre.
    """
    df = df.copy()
    # Define your conditions
    conditions = [
        ((df["Fav genre"] == df["Most_Listened_Genre"]) & (df["Number_Of_Most_Listened_Genre"] == 1)),
        ((df["Fav genre"] == df["Most_Listened_Genre"]) & (df["Number_Of_Most_Listened_Genre"] > 1)),
    ]

    # Define the results for each condition (matching the order above)
    choices = ["unique", True]

    # Apply using np.select(conditions, choices, default=False)
    df[ALIGNMENT] = np.select(conditions, choices, default=False)

    logger.info(f"{ALIGNMENT} sample (head):")
    logger.info("\n" + str(df[["Fav genre", "Most_Listened_Genre", ALIGNMENT]].head()))
    logger.info(f"{ALIGNMENT} counts:")
    logger.info("\n" + str(df[ALIGNMENT].value_counts()))
    return df

def summarize_alignment_distribution(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Q2 - Descriptive step:
    Summarize how common alignment is in the sample.

    We log:
    - counts of Alignment 
    - percentages of Alignment 
    """
    counts = df[ALIGNMENT].value_counts(dropna=False)
    percents = df[ALIGNMENT].value_counts(normalize=True, dropna=False) * 100

    logger.info(f"=== Q2: {ALIGNMENT} Distribution ===")
    logger.info("Counts (N):")
    logger.info("\n" + str(counts))

    logger.info("Percentages (%):")
    logger.info("\n" + str(percents.round(2)))


def compute_mental_health_index(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Create MENTAL_HEALTH_INDEX.
    We use the mean of Anxiety , Depression, Insomnia, and OCD for each participant.
    """
    df = df.copy()
    df[MENTAL_HEALTH_INDEX] = df[HEALTH_COLS].mean(axis=1)

    logger.info("Mental health columns + index (head):")
    logger.info("\n" + str(df[HEALTH_COLS + [MENTAL_HEALTH_INDEX]].head()))
    return df

def summarize_alignment_statistic(df, logger):
    """
    Descriptive statistics of Mental Health Index by Alignment.
    """
    summary = (
        df.groupby(ALIGNMENT)[MENTAL_HEALTH_INDEX]
        .agg(["count", "mean", "std"])
    )

    logger.info(f"=== Q2: {MENTAL_HEALTH_INDEX} by {ALIGNMENT} ===")
    logger.info(f"\n{summary}")

    return summary


def run_ttest(df: pd.DataFrame, outcome_col: str, logger: logging.Logger, label: str | None = None) -> tuple[float, float]:
    """
    Independent samples t-test comparing outcome_col between:
    - aligned participants
    - not aligned participants
    """
    aligned, unique_aligned = split_by_alignment(df, outcome_col)
    
    t_stat, p_value = ttest_ind(aligned, unique_aligned, nan_policy="omit")

    title = label or outcome_col
    logger.info(f"T-test results (Aligned vs Unique aligned) — {title}:")
    logger.info(f"t-statistic = {t_stat:.3f}")
    logger.info(f"p-value     = {p_value:.4f}")

    if p_value < 0.05:
        logger.info("Conclusion: Significant difference (p < 0.05). Reject H0.")
    else:
        logger.info("Conclusion: Not significant (p >= 0.05). Fail to reject H0.")

    return t_stat, p_value


def run_ttests_per_disorder(df: pd.DataFrame, health_cols: list[str], logger: logging.Logger) -> None:
    """
    Run independent t-tests for each mental health variable comparing aligned vs not-aligned.
    """
    logger.info("=== Q2: T-tests per Mental Health Variable ===")

    for col in health_cols:
        aligned, not_aligned = split_by_alignment(df, col)
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

    # 1) t-test on the overall index (this answers the main hypothesis directly)
    t_stat, p_value = run_ttest(df, MENTAL_HEALTH_INDEX, logger, label=MENTAL_HEALTH_INDEX)

    # 2) Visualize overall comparison
    plot_alignment_means(df, logger)
    plot_boxplot(df, logger)

    # 3) Build aggregated measure (Distress_Index) and test per final measures
    df = calculate_distress_index(df)
    run_ttests_per_disorder(df, AGGREGATED_HEALTH_COLS, logger)

    # 4) Visualize per-measure distributions (aligned vs not aligned)
    plot_disorders_by_alignment(df, AGGREGATED_HEALTH_COLS, logger)



    # -----------------------------
    # Interpretation (for submission)
    # -----------------------------
    logger.info("Interpretation:")
    if p_value < 0.05:
        logger.info(
            f"There is a statistically significant difference in {MENTAL_HEALTH_INDEX} "
            "between aligned and not-aligned participants in this sample."
        )
    else:
        logger.info(
            f"There is no statistically significant difference in {MENTAL_HEALTH_INDEX} "
            "between aligned and not-aligned participants in this sample."
        )
    
    
    logger.info("Done.")
  

 
