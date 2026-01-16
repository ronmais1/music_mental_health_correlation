def run_analysis():
    pass


#regression analysis code
# 1. Identify which genres belong to which cluster
cluster_names_map = {
    0: 'High Energy / Intensive',
    1: 'Acoustic / Relaxing',
    2: 'Electronic / Rhythmic'
}

# 2. Run clustering and get labels
genres, clusters = run_genre_clustering(df, genre_cols, cluster_names_map)

# 3. Create aggregated feature columns in the main dataframe
cluster_feature_names = []

for cluster_id, group_name in cluster_names_map.items():
    # Find all genres assigned to this cluster
    genres_in_cluster = [genres[i] for i in range(len(genres)) if clusters[i] == cluster_id]
    
    # Calculate the average frequency for this group per participant
    df[group_name] = df[genres_in_cluster].mean(axis=1)
    cluster_feature_names.append(group_name)
    
    logger.info(f"Created feature: {group_name} (based on {len(genres_in_cluster)} genres)")