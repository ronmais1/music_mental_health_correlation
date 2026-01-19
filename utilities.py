import pandas as pd
import logging
from pathlib import Path

def get_logger():
    """
    Initializes a basic logger for the project.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

def load_data(csv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Loads dataset from CSV path with existence check.
    """
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def basic_cleaning(df, logger, health_cols):
    """
    Cleans dataset by dropping NaNs in core research columns.
    """
    # Define required columns for a valid sample
    required = ["Age", "Hours per day"] + health_cols
    before = len(df)
    
    # Drop missing values and reset index
    df_clean = df.dropna(subset=required).copy()
    logger.info(f"Cleaning: {len(df_clean)} rows remaining (dropped {before - len(df_clean)})")
    return df_clean

def encode_categorical_data(df, columns, mapping):
    """
    Maps string frequency values to numerical scale (0-3).
    """
    df_encoded = df.copy()
    for col in columns:
        if col in df_encoded.columns:
            # Clean strings and apply numeric mapping
            df_encoded[col] = df_encoded[col].astype(str).str.strip().map(mapping).astype(float)
    return df_encoded

def get_descriptive_stats(df, columns, logger):
    """
    Logs mean, std, min, and max for specified columns.
    """
    stats = df[columns].describe().T
    logger.info(f"Descriptive Statistics:\n{stats[['mean', 'std', 'min', 'max']]}")
    return stats