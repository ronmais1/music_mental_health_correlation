import pandas as pd
import numpy as np
from consts import HEALTH_COLS, FREQ_MAPPING

def run_all_tests(logger):
    """
    Validation suite covering all major project stages: 
    Cleaning, Encoding, Feature Engineering, and Research Logic.
    """
    logger.info("="*45)
    logger.info("SYSTEM VALIDATION: TESTING ALL PROJECT STAGES")
    logger.info("="*45)

    # Creating Mock Data for testing assumptions
    mock_df = pd.DataFrame({
        'Age': [20, np.nan, 40],
        'Anxiety': [5, 10, 2],
        'Depression': [4, 8, 1],
        'Insomnia': [3, 5, 0],
        'OCD': [2, 4, 1],
        'Fav genre': ['Rock', 'Pop', 'Metal'],
        'Most listened genre': ['Rock', 'Jazz', 'Metal'],
        'Frequency [Rock]': ['Never', 'Sometimes', 'Very frequently']
    })

    # Stage 1: Data Cleaning Logic
    try:
        from utilities import basic_cleaning
        cleaned = basic_cleaning(mock_df, logger, HEALTH_COLS)
        # Verify that the row with NaN Age was removed
        assert len(cleaned) == 2
        logger.info("Stage 1: Data Cleaning Test - Passed")
    except Exception as e:
        logger.error(f"Stage 1: Data Cleaning Test - Failed: {e}")

    # Stage 2: Categorical Encoding Logic
    try:
        from utilities import encode_categorical_data
        encoded = encode_categorical_data(mock_df, ['Frequency [Rock]'], FREQ_MAPPING)
        # Verify 'Never' maps to 0
        assert encoded['Frequency [Rock]'].iloc[0] == 0
        logger.info("Stage 2: Categorical Encoding Test - Passed")
    except Exception as e:
        logger.error(f"Stage 2: Categorical Encoding Test - Failed: {e}")

    # Stage 3: Feature Engineering (Index Calculation)
    try:
        # Testing manual calculation of a mental health index
        mock_df['Health_Index'] = mock_df[HEALTH_COLS].mean(axis=1)
        # Expected average for the first row: (5+4+3+2)/4 = 3.5
        assert mock_df['Health_Index'].iloc[0] == 3.5
        logger.info("Stage 3: Feature Engineering Test - Passed")
    except Exception as e:
        logger.error(f"Stage 3: Feature Engineering Test - Failed: {e}")

    # Stage 4: Research Logic (Alignment for Question 2)
    try:
        # Testing the matching logic between favorite and most listened genres
        mock_df['Alignment'] = np.where(mock_df['Fav genre'] == mock_df['Most listened genre'], 'Matched', 'Mismatched')
        assert mock_df['Alignment'].iloc[0] == 'Matched'
        assert mock_df['Alignment'].iloc[1] == 'Mismatched'
        logger.info("Stage 4: Research Logic Test - Passed")
    except Exception as e:
        logger.error(f"Stage 4: Research Logic Test - Failed: {e}")

    logger.info("="*45 + "\n")