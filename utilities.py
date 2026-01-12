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