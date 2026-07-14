import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, ward, linkage
from scipy.spatial.distance import squareform
# from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale



def _average_pearson_correlation(features: pd.DataFrame) -> float:
    corr_matrix = features.corr().values
    if corr_matrix.shape[0] < 2:
        return np.nan
    upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
    corr_values = corr_matrix[upper_triangle_indices]
    if corr_values.size == 0:
        return np.nan
    corr_values = np.clip(corr_values, -1 + np.finfo(float).eps, 1 - np.finfo(float).eps)
    z_values = np.arctanh(corr_values)
    avg_z = np.mean(z_values)
    avg_corr = np.tanh(avg_z)
    return avg_corr



def detect_correlated_features(df, minimum_correlation = 0.8):
    if df.shape[1] < 2:
        print("Feature Clusters:")
        print("At least two features are required to detect correlations.")
        return {}

    # corr = spearmanr(df).correlation
    corr = (df).corr().values
    if np.isscalar(corr):
        corr = np.array([[1.0, corr], [corr, 1.0]], dtype=float)

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    # dist_linkage = ward(squareform(distance_matrix))
    dist_linkage = linkage(squareform(distance_matrix), method="average")

    thresholds = np.linspace(10.0, 0.05, 100)

    best_threshold = None
    best_clusters = {}
    # min_cluster_count = np.inf
    # min_cluster_count = -1
    max_total_features = -1

    for t in thresholds:
        cluster_labels = fcluster(dist_linkage, t, criterion="distance")
        df_tr = df.T.copy()
        # df_tr = X_stage.transpose().iloc[:,:-1].transpose().copy()
        df_tr['cluster'] = cluster_labels

        temp_clusters = {}
        valid = True

        for cl in np.unique(cluster_labels):
            columns = df_tr[df_tr['cluster'] == cl].index.tolist()
            if len(columns) < 2:
                continue  # skip singleton features
            avg_corr = _average_pearson_correlation(df[columns])
            if np.isfinite(avg_corr) and avg_corr >= minimum_correlation:
                sorted_cluster = tuple(sorted(columns))
                temp_clusters[sorted_cluster] = avg_corr
            else:
                valid = False
                break  # one bad cluster ruins this threshold

        if not valid or not temp_clusters:
            continue

        # here we are optimizing the number of clusters
        # maybe we should use the number of features per cluster as the criteria
        # this can lead to multiple tight clusters
        # if len(temp_clusters) > min_cluster_count:
        #     min_cluster_count = len(temp_clusters)
        #     best_threshold = t
        #     best_clusters = temp_clusters
        total_features = sum(len(cluster) for cluster in temp_clusters)

        if total_features > max_total_features:
            max_total_features = total_features
            best_threshold = t
            best_clusters = temp_clusters

    # Final result
    # print("Best threshold:", best_threshold)
    print("Feature Clusters:")
    if len(best_clusters.items()) == 0:
        print("No feature clusters were found!")
        return {}
    else:
        for features, corr in best_clusters.items():
            print(f"{features}: avg corr = {corr:.2f}")
        return best_clusters

def merge_correlated_features(df, correlated_clusters):
    if not correlated_clusters:
        return df.copy(deep=True)

    corr_features = {feature for cluster in correlated_clusters for feature in cluster}
    non_corr_features = [feature for feature in df.columns if feature not in corr_features]
    df_noncorr = df[non_corr_features].copy(deep=True)
    for i,fs in enumerate(correlated_clusters):
        pca = PCA(n_components=1)
        standardized_values = scale(df[list(fs)].values)
        X_transformed = pca.fit_transform(standardized_values)
        df_noncorr[f'PC_{i}'] = X_transformed.ravel()
    # df_noncorr['stage'] = df['stage2']
    return df_noncorr
