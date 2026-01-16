# music_and_mental_health.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
import logging
import numpy as np

# Configure logging to replace print commands
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Loading csv file
csv_path = "mxmh_survey_results.csv"
try:
    df = pd.read_csv(csv_path)
    logger.info("Dataset loaded successfully.")
except FileNotFoundError:
    raise SystemExit(f"CSV not found at {csv_path}")
except Exception as e:
    raise SystemExit(f"Failed to read CSV: {e}")

''' Dry preliminary data about the dataset '''

df.info(verbose=False) # Get a summary of the DataFrame structure
logger.info("\n")

# Classifying type of values
dtypes = df.dtypes.unique() # showing which columns belong to each data type
logger.info("Conclusion of data type:\n")
for dtype in dtypes:
    cols = df.select_dtypes(include=[dtype]).columns.tolist()
    logger.info(f"Data type: {dtype} ({len(cols)} columns)")
    logger.info(f"Columns: {', '.join(cols[:5])}...\n") # Showing first few columns
logger.info("\n")

# Showing basic data such as sample size, mean, std, etc.
basic_statistic_data = df.describe()
logger.info(f"{basic_statistic_data}\n")

# Finding the sum of missing values in every column
missing_values = df.isnull().sum()
logger.info("Missing values found in the following categories:\n")
for col in missing_values.index:
    if missing_values[col] > 0:
        logger.info(f"{col}: {missing_values[col]} missing values")
logger.info("\n")


''' Preparation for analysis, creating weighted index and mapping frequencies '''

# Plot 1: Heatmap of Mental Health disorders
health_cols = ["Anxiety", "Depression", "Insomnia", "OCD"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[health_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix: Mental Health Disorders")
plt.show()

# Create Weighted Distress Index (Anxiety + Depression)
df['Distress_Index'] = df[['Anxiety', 'Depression']].mean(axis=1)

# Define the targets to analyze
targets = {'Distress_Index': 'Anxiety & Depression', 'Insomnia': 'Insomnia', 'OCD': 'OCD'}

# Numerical Encoding of genre frequencies
# Never=0, Rarely=1, Sometimes=2, Very frequently=3
freq_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
genre_cols = [col for col in df.columns if col.startswith('Frequency [')]
for col in genre_cols:
    df[col] = df[col].apply(lambda x: freq_mapping.get(x.strip()) if isinstance(x, str) else x)

# Genre Clustering (Grouping genres based on listener patterns)
genre_data = df[genre_cols].dropna().astype(float).T

if not genre_data.empty:
    # 1. Run the clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    genre_clusters = kmeans.fit_predict(genre_data)
    
    # 2. Define the descriptive names you liked
    # Note: We assign names based on the cluster ID 
    cluster_names_map = {
        0: "High Energy / Intensive",
        1: "Acoustic / Relaxing",
        2: "Mainstream / Urban"
    }

    # 3. Prepare data for the graph (Grouped and Sorted)
    plot_df = pd.DataFrame({
        'Genre': genre_data.index,
        'Frequency': genre_data.mean(axis=1).values,
        'Cluster_ID': genre_clusters
    })
    plot_df['Cluster_Name'] = plot_df['Cluster_ID'].map(cluster_names_map)
    plot_df = plot_df.sort_values('Cluster_ID') # This ensures genres sit side-by-side

    # 4. Show the clustered bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Genre', y='Frequency', hue='Cluster_Name', palette='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title("Music Genres Grouped by Psychological Characteristics")
    plt.tight_layout()
    plt.show()

    # 5. Create the group variables for the regression
    clusters_dict = {i: [] for i in range(3)}
    for genre, cluster_id in zip(genre_data.index, genre_clusters):
        clusters_dict[cluster_id].append(genre)

    cluster_feature_names = []
    logger.info("\n--- Identified Genre Clusters ---")
    for cid, genres in clusters_dict.items():
        group_name = cluster_names_map[cid] # Using the descriptive names
        cluster_feature_names.append(group_name)
        
        # Calculate the average frequency for this group
        df[group_name] = df[genres].mean(axis=1)
        logger.info(f"Group '{group_name}' includes: {genres}")

else:
    logger.error("Error: Clustering failed.")
    raise SystemExit()

''' Main Analysis: Research Question Results '''

# Predictors for the full model
predictors = ['Age', 'Hours per day'] + cluster_feature_names

# Loop through each mental health target
for target_col, target_name in targets.items():
    analysis_df = df[[target_col] + predictors].dropna()
    y = analysis_df[target_col]
    
    # Control model: Age + Hours per day
    X_control = sm.add_constant(analysis_df[['Age', 'Hours per day']])
    model_control = sm.OLS(y, X_control).fit()
    
    # Full model: Age + Hours + Genre Clusters
    X_full = sm.add_constant(analysis_df[predictors])
    model_full = sm.OLS(y, X_full).fit()

    logger.info(f"\n--- Regression Results for {target_name} ---")
    logger.info(f"R² Control Model: {model_control.rsquared:.4f}")
    logger.info(f"R² Full Model: {model_full.rsquared:.4f}")
    logger.info(f"Unique Contribution of Music: {model_full.rsquared - model_control.rsquared:.4f}")
    logger.info(model_full.summary().as_text())

    # Plotting: Two-part chart showing R-squared gain and Coefficient significance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 2]})
    
    # Left Plot: Model Comparison (R-squared) - Proving the "Beyond Age and Hours" part
    ax1.bar(['Age & Hours', 'Full Model\n(+ Genres)'], [model_control.rsquared, model_full.rsquared], color=['#BDC3C7', '#5DADE2'])
    ax1.set_title(f"Prediction Power (R²) for {target_name}")
    ax1.set_ylabel("Variance Explained (R²)")
    gain = model_full.rsquared - model_control.rsquared
    ax1.annotate(f"Unique Gain:\n+{gain:.1%}", xy=(0.5, model_control.rsquared + (gain/2)), 
                 ha='center', fontweight='bold', color='black')

    # Right Plot: Regression Coefficients with Significance Stars
    coeffs = model_full.params[1:]
    pvals = model_full.pvalues[1:]
    
    # Defining stars for p-values
    def get_stars(p):
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return ''
    
    # Prepare labels and colors for the chart
    plot_labels = []
    plot_colors = []
    
    for i in range(len(coeffs)):
        p_val = pvals[i]
        # Get stars and add to the label
        stars = get_stars(p_val)
        plot_labels.append(coeffs.index[i] + " " + stars)
        
        # Simple color names: red for significant, gray for not
        if p_val < 0.05:
            plot_colors.append('red')
        else:
            plot_colors.append('gray')

    # Create the horizontal bar chart
    ax2.barh(plot_labels, coeffs.values, color=plot_colors)
    ax2.axvline(0, color='black', linestyle='--')
    ax2.set_title("Impact per Predictor (Adjusted) - " + target_name)
    ax2.set_xlabel("Effect Size")

    # Add a simple legend
    from matplotlib.lines import Line2D
    legend_list = [
        Line2D([0], [0], color='red', lw=4, label='Significant (p < 0.05)'),
        Line2D([0], [0], color='gray', lw=4, label='Not Significant'),
        Line2D([0], [0], color='white', label='* p<0.05, ** p<0.01, *** p<0.001')
    ]
    ax2.legend(handles=legend_list, loc='lower right')
    
    plt.tight_layout()
    plt.show()

logger.info("\nAnalysis Complete.")




