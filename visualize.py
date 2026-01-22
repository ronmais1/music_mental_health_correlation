import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D

def plot_correlation_heatmap(df, columns, logger):
    """
    Calculate and plot a correlation matrix for selected columns.
    """
    corr_matrix = df[columns].corr()
    logger.info("Correlation matrix calculated.")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Correlation Matrix: Mental Health Disorders")
    plt.tight_layout()
    plt.show()

def run_genre_clustering(df, genre_cols, cluster_names_map):
    """
    Groups music genres using K-Means and visualizes the results.
    """
    genre_data = df[genre_cols].dropna().astype(float).T
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    genre_clusters = kmeans.fit_predict(genre_data)
    
    plot_df = pd.DataFrame({
        'Genre': genre_data.index,
        'Frequency': genre_data.mean(axis=1).values,
        'Cluster_ID': genre_clusters
    })
    
    plot_df['Cluster_Name'] = plot_df['Cluster_ID'].map(cluster_names_map)
    plot_df = plot_df.sort_values('Cluster_ID')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Genre', y='Frequency', hue='Cluster_Name', palette='magma')
    
    plt.xticks(rotation=45, ha='right')
    plt.title("Music Genres Grouped by Psychological Characteristics")
    plt.tight_layout()
    plt.show()

    return genre_data.index, genre_clusters

def run_regression_analysis(df, predictors, targets, logger):
    """
    Runs OLS regression and plots dual-chart results with significance stars.
    """
    for target_col, target_name in targets.items():
        analysis_df = df[[target_col] + predictors].dropna()
        y = analysis_df[target_col]
        
        X_control = sm.add_constant(analysis_df[['Age', 'Hours per day']])
        model_control = sm.OLS(y, X_control).fit()
        
        X_full = sm.add_constant(analysis_df[predictors])
        model_full = sm.OLS(y, X_full).fit()

        logger.info(f"\n--- Regression Results for {target_name} ---")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 2]})
        
        # Left Plot: R-squared
        ax1.bar(['Age & Hours', 'Full Model\n(+ Genres)'], 
                [model_control.rsquared, model_full.rsquared], 
                color=['#BDC3C7', '#5DADE2'])
        ax1.set_title(f"Prediction Power (R²) for {target_name}")
        
        gain = model_full.rsquared - model_control.rsquared
        ax1.annotate(f"Unique Gain:\n+{gain:.1%}", xy=(0.5, model_control.rsquared + (gain/2)), 
                     ha='center', fontweight='bold', color='black')

        # Right Plot: Coefficients
        coeffs = model_full.params[1:]
        pvals = model_full.pvalues[1:]
        labels, colors = [], []
        
        for i in range(len(coeffs)):
            p = pvals[i]
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            labels.append(f"{coeffs.index[i]} {stars}")
            colors.append('red' if p < 0.05 else 'gray')

        ax2.barh(labels, coeffs.values, color=colors)
        ax2.axvline(0, color='black', linestyle='--')
        ax2.set_title(f"Impact per Predictor - {target_name}")

        legend_list = [
            Line2D([0], [0], color='red', lw=4, label='Significant (p < 0.05)'),
            Line2D([0], [0], color='gray', lw=4, label='Not Significant'),
            Line2D([0], [0], color='white', label='* p<0.05, ** p<0.01, *** p<0.001')
        ]
        ax2.legend(handles=legend_list, loc='lower right')
        
        plt.tight_layout()
        plt.show()

def plot_boxplot(df, logger):
    """
    Visualization: Boxplot for Mental Health Index by Music Alignment.
    """
    # Ensure columns exist before plotting to prevent crash
    if "Mental_Health_Index" not in df.columns or "Alignment" not in df.columns:
        logger.error("Required columns for boxplot are missing in DataFrame.")
        return

    plt.figure(figsize=(10, 6))
    df.boxplot(column="Mental_Health_Index", by="Alignment")

    plt.title("Mental Health Index by Music Alignment")
    plt.suptitle("") 
    plt.xlabel("Alignment (Favorite vs. Most Listened Genre)")
    plt.ylabel("Mental Health Index (0–10)")

    plt.tight_layout()
    plt.show()
    logger.info("Boxplot visualization displayed.")
    
def plot_alignment_means(df, logger):
    """
    Bar plot of mean Mental Health Index by Alignment with standard error.
    """
    stats = (
        df.groupby("Alignment")["Mental_Health_Index"]
        .agg(["mean", "std", "count"])
    )
    stats["se"] = stats["std"] / (stats["count"] ** 0.5)

    plt.figure(figsize=(8, 6))
    plt.bar(stats.index.astype(str), stats["mean"], yerr=stats["se"], capsize=5)
    plt.xlabel("Alignment")
    plt.ylabel("Mental Health Index")
    plt.title("Mean Mental Health Index by Alignment")
    plt.tight_layout()
    plt.show()

    logger.info("Mean comparison plot displayed.")
    
   
   def plot_disorders_by_alignment(df, health_cols, logger):
    """
    Boxplots for each mental health measure by Alignment (True/False).
    """
    if "Alignment" not in df.columns:
        logger.error("Column 'Alignment' is missing.")
        return

    long_df = (
        df.melt(
            id_vars=["Alignment"],
            value_vars=health_cols,
            var_name="Disorder",
            value_name="Score",
        )
        .dropna(subset=["Score"])
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=long_df, x="Disorder", y="Score", hue="Alignment")
    plt.title("Mental Health Scores by Alignment (per Disorder)")
    plt.xlabel("Mental Health Measure")
    plt.ylabel("Score (0–10)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    logger.info("Per-disorder boxplots displayed.")

