import pandas as pd
import numpy as np
import pytest
from utilities import basic_cleaning, encode_categorical_data
from consts import MENTAL_HEALTH_COLS, FREQ_MAPPING, AGE, ANXIETY, DEPRESSION

def test_data_cleaning_logic():
    """
    Test Case: Data Cleaning
    Verifies that the basic_cleaning function correctly removes rows 
    with missing values (NaN) in critical research columns like Age.
    """
    mock_df = pd.DataFrame({
        AGE: [25, np.nan, 30],
        'Hours per day': [2, 1, 5],
        ANXIETY: [5, 5, 5], 
        DEPRESSION: [5, 5, 5],
        'Insomnia': [5, 5, 5], 
        'OCD': [5, 5, 5]
    })
    
    # We pass None for the logger to simplify testing
    cleaned = basic_cleaning(mock_df, None, MENTAL_HEALTH_COLS)
    
    # Expectation: 1 row removed due to NaN Age, 2 rows remain.
    assert len(cleaned) == 2

def test_categorical_encoding():
    """
    Test Case: Categorical Encoding
    Ensures that frequency text labels (e.g., 'Never', 'Sometimes') 
    are correctly mapped to their corresponding numerical values (0-3).
    """
    mock_df = pd.DataFrame({'Frequency [Rock]': ['Never', 'Sometimes']})
    
    encoded = encode_categorical_data(mock_df, ['Frequency [Rock]'], FREQ_MAPPING, None)
    
    # Expectation: 'Never' maps to 0, 'Sometimes' maps to 2 based on FREQ_MAPPING
    assert encoded['Frequency [Rock]'].iloc[0] == 0
    assert encoded['Frequency [Rock]'].iloc[1] == 2

def test_mental_health_index_calculation():
    """
    Test Case: Feature Engineering
    Validates the mathematical calculation of the Mental Health Index 
    to ensure it correctly computes the mean across the core disorder columns.
    """
    mock_df = pd.DataFrame({
        ANXIETY: [10, 0],
        DEPRESSION: [10, 0],
        'Insomnia': [10, 0],
        'OCD': [10, 0]
    })
    
    # Calculate the mean across the mental health columns defined in consts
    mock_df['Mental_Health_Index'] = mock_df[MENTAL_HEALTH_COLS].mean(axis=1)
    
    # Expectation: Row 1 mean is 10.0, Row 2 mean is 0.0
    assert mock_df['Mental_Health_Index'].iloc[0] == 10.0
    assert mock_df['Mental_Health_Index'].iloc[1] == 0.0