Music & Mental Health Analysis Project
Project objectives
Assumptions
Thesis
Results

## Project Description
Main Objectives: To investigate the relationship between music listening habits, genre preferences, and mental health self-reports.

Question 1: Can music genre clusters predict mental health distress beyond demographic factors?

Question 2: Does the "Alignment" between a user's favorite genre and their most-listened-to genre impact their mental health index?

Hypotheses:

H1: Specific music genre clusters (e.g., High Intensity) will show different correlations with Anxiety and Depression.

H2: Participants with "Aligned" music preferences will report a lower Mental Health Index (better outcomes) than "Mismatched" participants.


## Project structure:
The project is built with a modular architecture to ensure clean code and reusability:

./main.py: Entry point. Orchestrates the execution of both research questions.
./genre_mental_correlation.py: Logic for Question 1 (Clustering & Regression).
./favourite_genre_to_mental_health.py: Logic for Question 2 (Alignment & T-Tests).
./utilities.py: Shared functions for data loading, cleaning, and logging.
./visualize.py: Centralized module for plotting (Heatmaps, Bar charts, Boxplots).
./consts.py: Centralized configuration, column names, and mappings.
./tests/test_final.py: Unit Tests. Comprehensive validation using pytest.



------------------------------------------------------------
## Key stages:
The project follows a standard Data Science workflow:

Data Import: Loading CSV via pathlib and pandas.

Data Processing: * Cleaning: Dropping invalid samples (NaNs in core columns).

Encoding: Mapping categorical frequency (Never...Very Frequently) to (0-3).

Modeling: * K-Means Clustering to group 16 genres into 3 psychological clusters.

OLS Regression to measure predictive power (R-squared).

Statistical Analysis: Performing independent samples T-tests for alignment.

Visualization: Generating diagnostic and result-oriented graphs (Seaborn/Matplotlib).

------------------------------------------------------------
## Important Definitions
Most Listened Genre: Identified by finding the highest frequency score across all genre columns per participant.

Alignment: A boolean variable indicating if Fav Genre == Most Listened Genre.

Distress Index: A mean score of Anxiety and Depression levels (0-10).

Mental Health Index: A composite mean of all 4 indicators (Anxiety, Depression, Insomnia, OCD).

Hypotheses:
H0 (null): mean Mental_Health_Index is the same in aligned and not-aligned participants.
H1 (alt) : mean Mental_Health_Index differs between aligned and not-aligned participants.

Statistical test:
Independent samples t-test (Aligned vs Not aligned), alpha = 0.05

Notes about interpretation:
- If p < 0.05 → reject H0 (significant difference)
- If p >= 0.05 → fail to reject H0 (no significant difference)
------------------------------------------------------------


##  Data Description
The dataset contains 736 responses from a survey about music and mental health.
Source: Music & Mental Health Survey Results (Kaggle)
https://www.google.com/search?q=https://www.kaggle.com/datasets/catherinayandaya/mxmh-survey-results

Categories: Demographics, Music Habits, Genre Frequencies (16 genres), and Mental Health scores.

## Instructions for running the project
Install Dependencies: pip install -r requirements.txt
Run Analysis: python main.py
Run Tests: python -m pytest
