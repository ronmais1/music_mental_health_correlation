import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from pathlib import Path
from utilities import load_data, basic_cleaning, encode_categorical_data, get_descriptive_stats
from visualize import run_genre_clustering, run_regression_analysis, plot_correlation_heatmap
from consts import MENTAL_HEALTH_COLS, FREQ_MAPPING, TARGET_GENRE_GROUPS, AGE, HOURS_PER_DAY, ANXIETY, DEPRESSION
import numpy as np

def run_question_one(logger):
    # 1. Load & Clean
    data_path = Path("mxmh_survey_results.csv")
    df = load_data(data_path, logger)
    df = basic_cleaning(df, logger, MENTAL_HEALTH_COLS)

    # 2. Pre-processing
    # Identify genre columns and convert to numbers
    genre_cols = [col for col in df.columns if col.startswith('Frequency [')]
    df = encode_categorical_data(df, genre_cols, FREQ_MAPPING, logger)

    # 3. Exploratory Analysis
    # Quick look at descriptive stats and correlations
    get_descriptive_stats(df, [AGE, HOURS_PER_DAY] + MENTAL_HEALTH_COLS, logger)
    plot_correlation_heatmap(df, MENTAL_HEALTH_COLS, logger)

    # 4. Clustering Phase (Question 1)
    cluster_names = {
        0: 'Heavy & Distortion-Based',
        1: 'Traditional & Acoustic',
        2: 'Urban & Electronic Beats'
    }
    
    genres, clusters = run_genre_clustering(df, genre_cols, cluster_names, logger)

    # Create the aggregated music features in the dataframe
    music_features = []
    for cid, name in cluster_names.items():
        genres_in_cluster = [genres[i] for i in range(len(genres)) if clusters[i] == cid]
        df[name] = df[genres_in_cluster].mean(axis=1)
        music_features.append(name)

    # 5. Regression Phase (Question 1)
    # Using our modular function with the turquoise/purple charts
    df['Distress_Index'] = df[[ANXIETY, DEPRESSION]].mean(axis=1)

    predictors = [AGE, HOURS_PER_DAY] + music_features
    run_regression_analysis(df, predictors, TARGET_GENRE_GROUPS, logger)