# Components/vector_quantization.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans on document vectors for coarse vector quantization."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return kmeans, kmeans.fit_predict(X)


def vq_query(
    query_vec: np.ndarray,
    X: np.ndarray,
    kmeans: KMeans,
    assignments: np.ndarray,
    top_n: int = 10,
    n_probes: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate nearest neighbors via clustering-based Vector Quantization."""
    query = query_vec.ravel()
    
    # Find n_probes closest clusters
    centroid_dists = euclidean_distances(query.reshape(1, -1), kmeans.cluster_centers_).ravel()
    n_probes = np.clip(n_probes, 1, len(centroid_dists))
    closest_clusters = np.argsort(centroid_dists)[:n_probes]
    
    # Get candidates from selected clusters
    candidate_idx = np.where(np.isin(assignments, closest_clusters))[0]
    if candidate_idx.size == 0:
        return np.array([], dtype=int), np.array([])
    
    # Compute exact distances and return top_n
    dists = euclidean_distances(query.reshape(1, -1), X[candidate_idx]).ravel()
    top_idx = np.argsort(dists)[:top_n]
    
    return candidate_idx[top_idx], dists[top_idx]