import pandas as pd
import numpy as np
import time
from Tokenizer import tokenize
from lsi_model import LSI
from lsh_index import LSHIndex, analyze_parameters
# ============================================================================
# STEP 1: Load Data and Create LSI Model
# ============================================================================
print("\n[1/5] Loading datasets...")
dataframes = []
for decade in ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']:
    df = pd.read_csv(f'../Data/{decade}-movies.csv')
    df['decade'] = decade
    dataframes.append(df)
    print(f"Loaded {len(df)} movies from {decade}")

all_movies = pd.concat(dataframes, ignore_index=True)
print(f"\nTotal movies: {len(all_movies)}")

# ============================================================================
# STEP 2: Tokenize and Build LSI Model
# ============================================================================
print("\n[2/5] Tokenizing and building LSI model...")
all_movies['tokens'] = all_movies.apply(
    lambda row: tokenize(str(row['title']) + ' ' + str(row['plot']), 
                        remove_stopwords=True, 
                        apply_stemming=True),
    axis=1
)

lsi = LSI(k=200)
token_lists = all_movies['tokens'].tolist()
lsi.fit(token_lists)
print("LSI model built!")

doc_vectors = lsi.doc_vectors.T 
print(f"Document vectors shape: {doc_vectors.shape}")

doc_vectors_normalized = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-10)

# ============================================================================
# STEP 3: Build LSH Index
# ============================================================================
print("\n[3/5] Building LSH Index...")

dimension = doc_vectors_normalized.shape[1]
k = 15
L = 10

print(f"\nParameter choices:")
print(f"  k = {k} (bits per hash code)")
print(f"  L = {L} (number of hash tables)")
print(f"  Expected buckets per table: 2^{k} = {2**k:,}")

# Build the index
lsh_index = LSHIndex(dimension=dimension, k=k, L=L)
lsh_index.index(doc_vectors_normalized)

# ============================================================================
# STEP 4: Test Queries
# ============================================================================
print("\n[4/5] Testing LSH Queries...")

test_movies = [
    "Star Wars",
    "The Godfather", 
    "Jurassic Park",
    "The Matrix"
]

for movie_title in test_movies:
    # Find the movie in dataset
    matches = all_movies[all_movies['title'].str.contains(movie_title, case=False, na=False)]
    
    if len(matches) == 0:
        print(f"\n'{movie_title}' not found in dataset")
        continue
    
    movie_idx = matches.index[0]
    movie = all_movies.iloc[movie_idx]
    
    print(f"\n" + "="*80)
    print(f"Query: {movie['title']} ({movie['decade']})")
    print(f"Plot: {movie['plot'][:150]}...")
    print("-"*80)
    
    # Get query vector
    query_vec = doc_vectors_normalized[movie_idx]
    
    # LSH search
    start_time = time.time()
    indices, scores = lsh_index.query(query_vec, top_k=5)
    lsh_time = time.time() - start_time
    
    # Get number of candidates examined
    n_candidates = lsh_index.get_candidate_count(query_vec)
    
    print(f"LSH Search:")
    print(f"  Time: {lsh_time*1000:.2f} ms")
    print(f"  Candidates examined: {n_candidates} out of {len(all_movies)} ({n_candidates/len(all_movies)*100:.1f}%)")
    print(f"\nTop 5 similar movies:")
    
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        similar_movie = all_movies.iloc[idx]
        print(f"  {rank}. [{score:.3f}] {similar_movie['title']} ({similar_movie['decade']})")
    
    # Compare with brute force
    start_time = time.time()
    brute_scores = (doc_vectors_normalized @ query_vec).flatten()
    brute_indices = np.argsort(-brute_scores)[:5]
    brute_time = time.time() - start_time
    
    print(f"\nBrute Force Search (for comparison):")
    print(f"  Time: {brute_time*1000:.2f} ms")
    print(f"  Speedup: {brute_time/lsh_time:.1f}x faster with LSH")
    print(f"\nTop 5 similar movies (brute force):")
    
    for rank, idx in enumerate(brute_indices, 1):
        similar_movie = all_movies.iloc[idx]
        score = brute_scores[idx]
        print(f"  {rank}. [{score:.3f}] {similar_movie['title']} ({similar_movie['decade']})")
    
    # Calculate recall
    lsh_set = set(indices)
    brute_set = set(brute_indices)
    recall = len(lsh_set & brute_set) / len(brute_set)
    print(f"\nRecall@5: {recall:.2%} (LSH found {len(lsh_set & brute_set)}/5 true nearest neighbors)")

# ============================================================================
# STEP 5: Parameter Analysis
# ============================================================================
print("\n" + "="*80)
print("[5/5] Parameter Analysis")
print("="*80)
print("Target: Find movies with cosine similarity ≥ 0.7")
print()

analyze_parameters(target_similarity=0.7, 
                  k_values=[10, 12, 15, 18, 20],
                  L_values=[5, 10, 15, 20])

print("\n" + "="*80)
print("Analysis Complete!")