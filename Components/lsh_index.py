import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


class LSHIndex:
    def __init__(self, dimension, k=15, L=10, random_seed=42):
        self.d = dimension
        self.k = k
        self.L = L
        self.random_seed = random_seed
        
        # Generate L sets of k random hyperplanes
        np.random.seed(random_seed)
        self.random_vectors = [
            np.random.randn(k, dimension) for _ in range(L)
        ]
        
        # Initialize L hash tables
        self.tables = [defaultdict(list) for _ in range(L)]
        
        self.vectors = None
        self.n_indexed = 0
        
    def _hash_vector(self, v, table_idx):
        random_vecs = self.random_vectors[table_idx]
        projections = random_vecs @ v
        hash_bits = (projections >= 0).astype(int)
        hash_value = 0
        for bit in hash_bits:
            hash_value = (hash_value << 1) | bit
            
        return hash_value
    
    def index(self, vectors):
        self.vectors = vectors
        self.n_indexed = len(vectors)
        
        print(f"Indexing {self.n_indexed} vectors with LSH...")
        print(f"  Dimension: {self.d}")
        print(f"  k (bits per hash): {self.k}")
        print(f"  L (hash tables): {self.L}")
        print(f"  Possible buckets per table: 2^{self.k} = {2**self.k:,}")
        
        # Hash each vector into all L tables
        for idx, v in enumerate(vectors):
            if idx % 5000 == 0 and idx > 0:
                print(f"  Indexed {idx}/{self.n_indexed} vectors...")
                
            for table_idx in range(self.L):
                h = self._hash_vector(v, table_idx)
                self.tables[table_idx][h].append(idx)
        
        print(f"Indexing complete!")
        self._print_statistics()
    
    def _print_statistics(self):
        print(f"\nLSH Index Statistics:")
        for table_idx in range(self.L):
            n_buckets = len(self.tables[table_idx])
            bucket_sizes = [len(bucket) for bucket in self.tables[table_idx].values()]
            avg_size = np.mean(bucket_sizes) if bucket_sizes else 0
            max_size = max(bucket_sizes) if bucket_sizes else 0
            
            print(f"  Table {table_idx+1}: {n_buckets} non-empty buckets, "
                  f"avg size: {avg_size:.1f}, max size: {max_size}")
    
    def query(self, query_vec, top_k=10, return_distances=True):
        """
        Find approximate nearest neighbors using LSH.
        Args:
            query_vec: Query vector (d-dimensional)
            top_k: Number of results to return
            return_distances: If True, return (indices, scores), else just indices
            
        Returns:
            If return_distances=True: (indices, cosine_similarities)
            If return_distances=False: indices only
        """
        if self.vectors is None:
            raise ValueError("Index not built. Call index() first.")
        
        # Collect candidates from all L tables
        candidates = set()
        
        for table_idx in range(self.L):
            h = self._hash_vector(query_vec, table_idx)
            if h in self.tables[table_idx]:
                candidates.update(self.tables[table_idx][h])
        
        # Handle case where no candidates found
        if not candidates:
            print("Warning: No candidates found in hash tables. Returning empty result.")
            return (np.array([], dtype=int), np.array([])) if return_distances else np.array([], dtype=int)
        
        # Convert to list for indexing
        candidate_indices = np.array(list(candidates))
        
        # Re-rank candidates using exact cosine similarity
        candidate_vectors = self.vectors[candidate_indices]
        similarities = cosine_similarity(
            query_vec.reshape(1, -1), 
            candidate_vectors
        ).flatten()
        
        # Get top-k
        top_k = min(top_k, len(candidate_indices))
        top_positions = np.argsort(-similarities)[:top_k]
        
        result_indices = candidate_indices[top_positions]
        result_scores = similarities[top_positions]
        
        if return_distances:
            return result_indices, result_scores
        else:
            return result_indices
    
    def batch_query(self, query_vectors, top_k=10):
        """
        Query multiple vectors at once.
        """
        results = []
        for q in query_vectors:
            results.append(self.query(q, top_k=top_k))
        return results
    
    def get_candidate_count(self, query_vec):
        candidates = set()
        for table_idx in range(self.L):
            h = self._hash_vector(query_vec, table_idx)
            if h in self.tables[table_idx]:
                candidates.update(self.tables[table_idx][h])
        return len(candidates)


def estimate_collision_probability(similarity, k):
    """
    Estimate probability that two vectors with given cosine similarity
    will hash to the same k-bit code.
    
    Theory:
        p = (1 - θ/π)^k
        where θ = arccos(similarity)
    
    Args:
        similarity: Cosine similarity (0 to 1)
        k: Number of hash functions
        
    Returns:
        Probability of collision
    """
    theta = np.arccos(np.clip(similarity, -1, 1))
    p_single = 1 - theta / np.pi
    return p_single ** k


def estimate_recall(similarity, k, L):
    """
    Estimate recall (probability of finding similar item) for given LSH parameters.
    
    Theory:
        P(found in ≥1 table) = 1 - (1 - p^k)^L
        where p = 1 - arccos(similarity)/π
    """
    p_single = 1 - np.arccos(np.clip(similarity, -1, 1)) / np.pi
    p_table = p_single ** k
    recall = 1 - (1 - p_table) ** L
    return recall


def analyze_parameters(target_similarity=0.7, k_values=None, L_values=None):
    """
    Analyze how LSH parameters affect recall for a target similarity.
    """
    if k_values is None:
        k_values = [5, 10, 15, 20, 25]
    if L_values is None:
        L_values = [1, 5, 10, 15, 20]
    
    print(f"LSH Parameter Analysis for similarity = {target_similarity}")
    print("="*70)
    print(f"{'k (bits)':>10} {'L (tables)':>12} {'Recall':>12} {'Buckets':>15}")
    print("-"*70)
    
    for k in k_values:
        for L in L_values:
            recall = estimate_recall(target_similarity, k, L)
            buckets = 2**k
            print(f"{k:>10} {L:>12} {recall:>12.4f} {buckets:>15,}")
    print("="*70)