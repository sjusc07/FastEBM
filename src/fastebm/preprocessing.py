import numpy as np
import scipy
import pandas as pd
import sklearn



def _average_pearson_correlation(features: pd.DataFrame) -> float:
    corr_matrix = features.corr().values
    upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
    corr_values = corr_matrix[upper_triangle_indices]
    z_values = np.arctanh(corr_values)
    avg_z = np.mean(z_values)
    avg_corr = np.tanh(avg_z)
    return avg_corr



def detect_correlated_features(df, minimum_correlation = 0.8):
    corr = scipy.stats.spearmanr(df).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = scipy.cluster.hierarchy.ward(scipy.spatial.distance.squareform(distance_matrix))

    thresholds = np.linspace(10.0, 0.05, 100)
    minimum_correlation = 0.8          ## the minimum value to consider features to be a correlated cluster

    best_threshold = None
    best_clusters = {}
    # min_cluster_count = np.inf
    # min_cluster_count = -1
    max_total_features = -1

    for t in thresholds:
        cluster_labels = scipy.cluster.hierarchy.fcluster(dist_linkage, t, criterion="distance")
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
            if avg_corr >= minimum_correlation:
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
        return None
    else:
        for features, corr in best_clusters.items():
            print(f"{features}: avg corr = {corr:.2f}")
        return best_clusters

def merge_correlated_features(df, correlated_clusters):
    corr_features = set(list(sum( list(correlated_clusters.keys()), ())))
    all_features = set(df.columns)
    # non_corr_features = (all_features - corr_features) - set(['stage2'])
    non_corr_features = (all_features - corr_features)
    # non_corr_features = 
    df_noncorr = df[list(non_corr_features)].copy(deep=True)
    for i,fs in enumerate(list(correlated_clusters.keys())):
        pca = sklearn.decomposition.PCA(n_components=1)
        pca.fit(sklearn.preprocessing.scale(df[list(fs)].values))
        X_transformed = pca.transform(sklearn.preprocessing.scale(df[list(fs)].values))
        df_noncorr[f'PC_{i}'] = X_transformed
    # df_noncorr['stage'] = df['stage2']
    return df_noncorr
