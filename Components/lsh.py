# Components/lsh.py
"""
Locality-Sensitive Hashing (LSH) for cosine similarity over dense document vectors.

We follow the lecture notation:
- m  = n_hashes        (number of hash functions / rows of the signature)
- b  = n_bands         (number of bands)
- r  = rows_per_band   (rows per band), with m = b * r

Pipeline:
1) Generate m random hyperplanes in R^d
2) Compute binary signatures sign(X * W^T) in {0,1}^m
3) Partition signatures into b bands of r bits
4) For each band, build a hash table: band_key -> list of doc indices
5) For a query:
   - compute its signature
   - collect candidate docs from all matching buckets
   - re-rank candidates using exact cosine similarity
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Set


# ----------------------------------------------------------------------
# 1. Hyperplanes and signatures
# ----------------------------------------------------------------------

def generate_hyperplanes(
    dim: int,
    n_hashes: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate random hyperplanes for cosine LSH.

    Each hash h_j(x) = sign(w_j · x), where w_j is a random vector.

    Args:
        dim: dimensionality of the document vectors (d).
        n_hashes: number of hash functions m (rows of the signature).
        random_state: seed for reproducibility.

    Returns:
        hyperplanes: array of shape (m, d).
    """
    rng = np.random.default_rng(random_state)
    # Standard normal entries → random directions in R^d
    hyperplanes = rng.standard_normal(size=(n_hashes, dim))
    return hyperplanes


def compute_signatures(
    X: np.ndarray,
    hyperplanes: np.ndarray,
) -> np.ndarray:
    """
    Compute LSH binary signatures for all documents.

    Signature matrix S has shape (n_docs, m) with entries in {0,1}.
    S[i, j] = 1 if hyperplane j classifies x_i on the positive side, else 0.

    Args:
        X: document matrix of shape (n_docs, dim).
        hyperplanes: array of shape (m, dim).

    Returns:
        signatures: uint8 array of shape (n_docs, m) with bits in {0,1}.
    """
    # Shape: (n_docs, m)
    projections = X @ hyperplanes.T
    # sign > 0 → 1, else 0
    signatures = (projections >= 0).astype(np.uint8)
    return signatures


# ----------------------------------------------------------------------
# 2. Building the LSH index (bands + hash tables)
# ----------------------------------------------------------------------

def build_lsh_index(
    signatures: np.ndarray,
    n_bands: int,
) -> Dict:
    """
    Build the LSH index using banding as in the lecture.

    Args:
        signatures: array of shape (n_docs, m) with bits in {0,1}.
        n_bands: number of bands b.

    Returns:
        index: dictionary containing:
            - "n_bands": b
            - "rows_per_band": r
            - "band_tables": list of length b
              where band_tables[i] is a dict:
                  band_key (tuple of bits) -> list of doc indices
    """
    n_docs, m = signatures.shape
    assert m % n_bands == 0, "n_hashes (m) must be divisible by n_bands (b)."

    rows_per_band = m // n_bands  # r
    band_tables: List[Dict[Tuple[int, ...], List[int]]] = [
        {} for _ in range(n_bands)
    ]

    for doc_idx in range(n_docs):
        sig = signatures[doc_idx]
        # For each band, take the slice of r bits
        for b in range(n_bands):
            start = b * rows_per_band
            end = (b + 1) * rows_per_band
            band_bits = tuple(sig[start:end])  # small, hashable

            table = band_tables[b]
            if band_bits not in table:
                table[band_bits] = []
            table[band_bits].append(doc_idx)

    index = {
        "n_bands": n_bands,
        "rows_per_band": rows_per_band,
        "band_tables": band_tables,
    }
    return index


# ----------------------------------------------------------------------
# 3. Candidate generation for a query
# ----------------------------------------------------------------------

def lsh_candidates(
    query_signature: np.ndarray,
    index: Dict,
) -> np.ndarray:
    """
    Return the set of candidate document indices for a given query signature.

    Strategy:
      - For each band b:
          * take the r bits of the query for that band,
          * look up the corresponding bucket,
          * add all doc indices in that bucket to the candidate set.

    Args:
        query_signature: array of shape (m,) with bits in {0,1}.
        index: dictionary created by build_lsh_index().

    Returns:
        candidates: 1D array of unique document indices (order not guaranteed).
    """
    n_bands = index["n_bands"]
    rows_per_band = index["rows_per_band"]
    band_tables = index["band_tables"]

    candidates: Set[int] = set()
    m = query_signature.shape[0]
    assert m == n_bands * rows_per_band, "Query signature length mismatch."

    for b in range(n_bands):
        start = b * rows_per_band
        end = (b + 1) * rows_per_band
        band_bits = tuple(query_signature[start:end])

        table = band_tables[b]
        if band_bits in table:
            for doc_idx in table[band_bits]:
                candidates.add(doc_idx)

    if not candidates:
        return np.array([], dtype=int)

    return np.fromiter(candidates, dtype=int)


# ----------------------------------------------------------------------
# 4. Cosine similarity + final top-k query
# ----------------------------------------------------------------------

def _cosine_similarities(
    query_vec: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between query_vec and all rows of X.

    Args:
        query_vec: array of shape (dim,).
        X: array of shape (n_docs, dim).

    Returns:
        sims: array of shape (n_docs,) with cosine similarities in [-1, 1].
    """
    if query_vec.ndim != 1:
        query_vec = query_vec.ravel()

    # Normalize query and documents
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0.0:
        raise ValueError("Query vector has zero norm.")
    query_unit = query_vec / q_norm

    X_norms = np.linalg.norm(X, axis=1)
    # Avoid division by zero: mask zero-norm rows
    valid = X_norms > 0
    sims = np.zeros(X.shape[0], dtype=np.float32)
    sims[valid] = (X[valid] @ query_unit) / X_norms[valid]
    return sims


def lsh_query(
    query_vec: np.ndarray,
    doc_vectors: np.ndarray,
    hyperplanes: np.ndarray,
    index: Dict,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full LSH query pipeline for cosine similarity.

    Steps:
      1) Compute LSH signature of the query using the same hyperplanes.
      2) Use the LSH index to obtain a set of candidate documents.
      3) Compute exact cosine similarities on candidates only.
      4) Return top_k documents sorted by similarity (descending).

    Args:
        query_vec: array of shape (dim,).
        doc_vectors: array of shape (n_docs, dim).
        hyperplanes: array of shape (m, dim) used for indexing.
        index: LSH index dictionary from build_lsh_index().
        top_k: number of neighbors to return.

    Returns:
        top_indices: array of document indices (subset of [0..n_docs-1]).
        top_sims: array of cosine similarities corresponding to top_indices.
    """
    # 1) Signature of the query
    proj = hyperplanes @ query_vec  # shape (m,)
    query_sig = (proj >= 0).astype(np.uint8)

    # 2) Candidate set from LSH
    cand_idx = lsh_candidates(query_sig, index)
    if cand_idx.size == 0:
        # No candidates: return empty results
        return cand_idx, np.array([], dtype=np.float32)

    # 3) Cosine similarities restricted to candidates
    sims = _cosine_similarities(query_vec, doc_vectors[cand_idx])

    # 4) Top-k ranking
    k = min(top_k, cand_idx.size)
    order = np.argsort(-sims)[:k]  # sort by similarity descending
    return cand_idx[order], sims[order]


# ----------------------------------------------------------------------
# 5. Convenience builder: from raw document vectors to full LSH index
# ----------------------------------------------------------------------

def build_lsh_from_vectors(
    X: np.ndarray,
    n_hashes: int,
    n_bands: int,
    random_state: int = 42,
) -> Dict:
    """
    High-level helper: from dense vectors to a complete LSH structure.

    This wraps steps:
        1) generate hyperplanes,
        2) compute signatures,
        3) build the band hash tables.

    Args:
        X: document matrix of shape (n_docs, dim).
        n_hashes: total number of hash functions m.
        n_bands: number of bands b (must divide m).
        random_state: seed.

    Returns:
        lsh_struct: dictionary with:
            - "hyperplanes": (m, dim) array
            - "signatures": (n_docs, m) array
            - "index": LSH index from build_lsh_index()
    """
    dim = X.shape[1]
    hyperplanes = generate_hyperplanes(dim=dim, n_hashes=n_hashes, random_state=random_state)
    signatures = compute_signatures(X, hyperplanes)
    index = build_lsh_index(signatures, n_bands=n_bands)

    return {
        "hyperplanes": hyperplanes,
        "signatures": signatures,
        "index": index,
    }
