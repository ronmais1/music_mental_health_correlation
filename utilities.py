from pathlib import Path
import logging
import pandas as pd
from consts import HEALTH_COLS

def load_data(csv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Using Path makes the code robust to different working directories.
    If the file does not exist in the expected location, raise a clear error.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def basic_cleaning(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    We drop rows with missing values in the columns required for this research question.
    """
    required = ["Age", "Hours per day"] + HEALTH_COLS
    before = df.shape[0]
    df = df.dropna(subset=required).copy()
    after = df.shape[0]
    logger.info(f"Rows after cleaning: {after} (dropped {before - after})")
    return df

def get_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def encode_categorical_data(df, columns, mapping):
    """
    Standardizes categorical string values into numerical format 
    based on a provided mapping dictionary.
    """
    # Create a copy to avoid modifying the original dataframe unexpectedly
    df_encoded = df.copy()
    
    for col in columns:
        if col in df_encoded.columns:
            # Clean string whitespace and map values
            # Using map is more efficient than apply(lambda) for dictionaries
            df_encoded[col] = df_encoded[col].astype(str).str.strip().map(mapping)
            
            # Convert to float to handle potential NaN values gracefully
            df_encoded[col] = df_encoded[col].astype(float)
            
    return df_encoded

def check_missing_data(df, logger, threshold=15.0):
    """
    Check for missing values and warn if above threshold.
    """
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    # Filter only columns that actually have missing values
    stats = missing_pct[missing_pct > 0]
    
    if stats.empty:
        logger.info("No missing values.")
        return stats

    for col, pct in stats.items():
        msg = f"{col}: {pct:.2f}% missing"
        if pct > threshold:
            logger.warning(f"HIGH MISSING DATA: {msg}")
        else:
            logger.info(msg)
            
    return stats

def get_descriptive_stats(df, columns, logger):
    """
    Log descriptive statistics for specific research columns.
    """
    stats = df[columns].describe().T
    logger.info(f"Descriptive Statistics:\n{stats[['mean', 'std', 'min', 'max']]}")
    return stats