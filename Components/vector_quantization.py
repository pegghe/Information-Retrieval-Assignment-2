# Components/vector_quantization.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import euclidean_distances


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42):
    """
    Fit KMeans on document vectors.

    Args:
        X: array (n_docs, dim)
        k: number of clusters
        random_state: seed

    Returns:
        kmeans: trained KMeans model
        assignments: array (n_docs,) with cluster index per document
    """
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init="auto"
    )
    assignments = kmeans.fit_predict(X)
    return kmeans, assignments


def vq_query(query_vec: np.ndarray,
             X: np.ndarray,
             kmeans: KMeans,
             assignments: np.ndarray,
             top_n: int = 10):
    """
    Approximate NN via Vector Quantization:
    - find closest centroid
    - restrict search to that cluster

    Args:
        query_vec: (dim,)
        X: full matrix (n_docs, dim)
        kmeans: trained model
        assignments: cluster for each doc
        top_n: neighbors to return

    Returns:
        indices: indices in X of nearest docs
        distances: distances to query
    """
    # nearest centroid
    closest_cluster, _ = pairwise_distances_argmin_min(
        query_vec.reshape(1, -1),
        kmeans.cluster_centers_
    )
    c = int(closest_cluster[0])

    # docs in that cluster
    idx = np.where(assignments == c)[0]
    if idx.size == 0:
        return np.array([], dtype=int), np.array([])

    X_cluster = X[idx]
    dists = euclidean_distances(
        query_vec.reshape(1, -1),
        X_cluster
    ).ravel()

    order = np.argsort(dists)[:top_n]
    return idx[order], dists[order]
