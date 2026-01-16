import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import statsmodels.api as sm


def plot_correlation_heatmap(df, columns, logger):
    """
    Calculate and plot a correlation matrix for selected columns.
    """

    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    logger.info("Correlation matrix calculated.")

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Variable Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    return corr_matrix

def run_genre_clustering(df, genre_cols, cluster_names_map):
    """
    Groups music genres using K-Means and visualizes the results.
    Maintains the original sorting and 'magma' color palette.
    """

    # Step 1: Data preparation (Transposing for genre-based clustering)
    genre_data = df[genre_cols].dropna().astype(float).T
    
    # Step 2: Clustering execution
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    genre_clusters = kmeans.fit_predict(genre_data)
    
    # Step 3: Formatting for visualization
    plot_df = pd.DataFrame({
        'Genre': genre_data.index,
        'Frequency': genre_data.mean(axis=1).values,
        'Cluster_ID': genre_clusters
    })
    
    # Mapping cluster names and sorting by ID
    plot_df['Cluster_Name'] = plot_df['Cluster_ID'].map(cluster_names_map)
    plot_df = plot_df.sort_values('Cluster_ID')

    # Step 4: Visualizing the clustered bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Genre', y='Frequency', hue='Cluster_Name', palette='magma')
    
    plt.xticks(rotation=45, ha='right')
    plt.title("Music Genres Grouped by Psychological Characteristics")
    plt.tight_layout()
    plt.show()

    # Returning results for the next steps in analysis
    return genre_data.index, genre_clusters



def run_regression_analysis(df, predictors, targets, logger):
    """
    Runs OLS regression and plots results with significance stars.
    """
    for col, name in targets.items():
        # Prepare data
        data = df[[col] + predictors].dropna()
        y = data[col]
        
        # Models
        X_base = sm.add_constant(data[['Age', 'Hours per day']])
        res_base = sm.OLS(y, X_base).fit()
        
        X_full = sm.add_constant(data[predictors])
        res_full = sm.OLS(y, X_full).fit()
        
        gain = res_full.rsquared - res_base.rsquared
        logger.info(f"{name} - R2 Gain: {gain:.4f}")

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 2]})
        
        # R2 Comparison
        ax1.bar(['Base', 'Full'], [res_base.rsquared, res_full.rsquared], color=['#ecf0f1', '#3498db'])
        ax1.set_title(f"R-squared: {name}")
        
        # Coefficients & Stars
        coeffs = res_full.params[1:]
        pvals = res_full.pvalues[1:]
        labels = []
        colors = []
        
        for i in range(len(coeffs)):
            p = pvals[i]
            # Significance stars logic
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            labels.append(f"{coeffs.index[i]} {stars}")
            colors.append('#e74c3c' if p < 0.05 else '#bdc3c7')

        ax2.barh(labels, coeffs.values, color=colors)
        ax2.axvline(0, color='black', ls='--')
        ax2.set_title(f"Predictor Impact: {name}")
        
        plt.tight_layout()
        plt.show()