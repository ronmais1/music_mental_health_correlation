import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind

def _apply_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _alignment_labels():
    order = [False, True]
    labels = {False: "Not aligned", True: "Aligned"}
    return order, labels


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
    _apply_plot_style()
    order, labels = _alignment_labels()

    if "Mental_Health_Index" not in df.columns or "Alignment" not in df.columns:
        logger.error("Required columns for boxplot are missing in DataFrame.")
        return

    plot_df = df.dropna(subset=["Mental_Health_Index", "Alignment"]).copy()
    plot_df["Alignment_Label"] = plot_df["Alignment"].map(labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=plot_df,
        x="Alignment_Label",
        y="Mental_Health_Index",
        order=[labels[False], labels[True]],
        palette="magma",
        width=0.5,
        ax=ax
    )
    sns.stripplot(
        data=plot_df,
        x="Alignment_Label",
        y="Mental_Health_Index",
        order=[labels[False], labels[True]],
        color="black",
        alpha=0.25,
        size=3,
        jitter=0.2,
        ax=ax
    )

    ax.set_title("Mental Health Index distribution by Music Alignment")
    ax.set_xlabel("Alignment (Favorite vs. Most listened genre)")
    ax.set_ylabel("Mental Health Index (0–10)")
    ax.set_ylim(0, 10)

    plt.tight_layout()
    plt.show()
    logger.info("Boxplot visualization displayed.")


def plot_alignment_means(df, logger):
    _apply_plot_style()
    order, labels = _alignment_labels()

    if "Mental_Health_Index" not in df.columns or "Alignment" not in df.columns:
        logger.error("Required columns for plot_alignment_means are missing.")
        return

    plot_df = df.dropna(subset=["Mental_Health_Index", "Alignment"]).copy()
    plot_df["Alignment_Label"] = plot_df["Alignment"].map(labels)

    stats = (
        plot_df.groupby("Alignment_Label")["Mental_Health_Index"]
        .agg(["mean", "std", "count"])
        .reindex([labels[False], labels[True]])
    )
    stats["se"] = stats["std"] / (stats["count"] ** 0.5)
    stats["ci95"] = 1.96 * stats["se"]

    aligned = plot_df.loc[plot_df["Alignment"] == True, "Mental_Health_Index"]
    not_aligned = plot_df.loc[plot_df["Alignment"] == False, "Mental_Health_Index"]
    t_stat, p_value = ttest_ind(aligned, not_aligned, nan_policy="omit")

    fig, ax = plt.subplots(figsize=(9, 6))

    x = stats.index.tolist()
    y = stats["mean"].values
    yerr = stats["ci95"].values

    sns.barplot(x=x, y=y, hue=x, ax=ax, palette="magma", errorbar=None,legend=False)
    ax.errorbar(range(len(x)), y, yerr=yerr, fmt="none", capsize=6)

    ax.set_title("Mean Mental Health Index by Music Alignment")
    ax.set_xlabel("Alignment (Favorite vs. Most listened genre)")
    ax.set_ylabel("Mental Health Index (0–10)")
    ax.set_ylim(0, 10)

    for i, grp in enumerate(x):
        n = int(stats.loc[grp, "count"])
        ax.text(i, y[i] + yerr[i] + 0.2, f"n={n}", ha="center", va="bottom", fontsize=12)

    ax.text(0.5, 0.02, f"t = {t_stat:.3f} | p = {p_value:.4f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=12)

    plt.tight_layout()
    plt.show()
    logger.info("Mean comparison plot displayed.")

   
def plot_disorders_by_alignment(df, health_cols, logger):
    _apply_plot_style()
    order, labels = _alignment_labels()

    if "Alignment" not in df.columns:
        logger.error("Column 'Alignment' is missing.")
        return

    pretty = {
        "Distress_Index": "Distress Index\n(Anxiety+Depression)",
        "Insomnia": "Insomnia",
        "OCD": "OCD",
        "Anxiety": "Anxiety",
        "Depression": "Depression",
    }

    use_cols = [c for c in health_cols if c in df.columns]
    if not use_cols:
        logger.error("None of the requested health_cols exist in the DataFrame.")
        return

    plot_df = df[["Alignment"] + use_cols].dropna(subset=["Alignment"]).copy()
    plot_df["Alignment_Label"] = plot_df["Alignment"].map(labels)

    long_df = plot_df.melt(
        id_vars=["Alignment_Label"],
        value_vars=use_cols,
        var_name="Measure",
        value_name="Score"
    ).dropna(subset=["Score"])

    long_df["Measure"] = long_df["Measure"].map(lambda x: pretty.get(x, x))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=long_df,
        x="Measure",
        y="Score",
        hue="Alignment_Label",
        palette="magma",
        ax=ax,
        legend=False
    )

    ax.set_title("Mental health measures by Alignment")
    ax.set_xlabel("")
    ax.set_ylabel("Score (0–10)")
    ax.set_ylim(0, 10)
    ax.legend(title="Alignment", loc="upper right")

    plt.tight_layout()
    plt.show()
    logger.info("Per-measure boxplots displayed.")

