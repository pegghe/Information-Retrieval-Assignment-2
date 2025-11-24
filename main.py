"""
High-Dimensional Search 
"""

import pandas as pd
import sys
sys.path.append('Components')
from Tokenizer import tokenize
from lsi_model import LSI

print("="*80)
print("Wikipedia Movies Retrieval System")
print("="*80)

# Load all movie datasets
print("\n[1/5] Loading datasets...")
dataframes = []
for decade in ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']:
    df = pd.read_csv(f'data/{decade}-movies.csv')
    df['decade'] = decade
    dataframes.append(df)
    print(f"  ✓ Loaded {len(df)} movies from {decade}")

# Combine all movies
all_movies = pd.concat(dataframes, ignore_index=True)
print(f"\nTotal movies loaded: {len(all_movies)}")

# Data exploration
print("\n[2/5] Data Exploration...")
print("-"*80)
print("\nDataset Info:")
print(f"  - Columns: {list(all_movies.columns)}")
print(f"  - Shape: {all_movies.shape}")
print(f"  - Missing values: {all_movies.isnull().sum().to_dict()}")

print("\n\nFirst 3 movies:")
print("-"*80)
for i in range(3):
    movie = all_movies.iloc[i]
    print(f"\n{i+1}. {movie['title']} ({movie['decade']})")
    plot_preview = movie['plot'][:150] + "..." if len(movie['plot']) > 150 else movie['plot']
    print(f"   Plot: {plot_preview}")

print("\n\nMovies per decade:")
print(all_movies['decade'].value_counts().sort_index())

print("\n\nPlot length statistics:")
all_movies['plot_length'] = all_movies['plot'].str.len()
print(f"  - Mean: {all_movies['plot_length'].mean():.0f} characters")
print(f"  - Median: {all_movies['plot_length'].median():.0f} characters")
print(f"  - Min: {all_movies['plot_length'].min():.0f} characters")
print(f"  - Max: {all_movies['plot_length'].max():.0f} characters")

# Tokenize all documents
print("\n[3/5] Tokenizing documents...")
print("Processing movie titles and plots...")
print("Using NLTK tokenization with stemming (Porter) for optimal IR performance...")

# Tokenize each movie (title + plot)
# For indexing: use stemming and remove stopwords
all_movies['tokens'] = all_movies.apply(
    lambda row: tokenize(str(row['title']) + ' ' + str(row['plot']), 
                        remove_stopwords=True, 
                        apply_stemming=True),
    axis=1
)

# Count total tokens
total_tokens = sum(len(tokens) for tokens in all_movies['tokens'])
print(f"  ✓ Processed {len(all_movies)} documents")
print(f"  ✓ Total tokens: {total_tokens:,}")

# Tokenization analysis
print("\n[4/5] Tokenization Analysis...")
print("-"*80)

# Token count per document
all_movies['token_count'] = all_movies['tokens'].apply(len)
print(f"\nTokens per document statistics:")
print(f"  - Mean: {all_movies['token_count'].mean():.1f} tokens")
print(f"  - Median: {all_movies['token_count'].median():.1f} tokens")
print(f"  - Min: {all_movies['token_count'].min()} tokens")
print(f"  - Max: {all_movies['token_count'].max()} tokens")

# Build vocabulary
print(f"\nBuilding vocabulary...")
vocabulary = set()
for tokens in all_movies['tokens']:
    vocabulary.update(tokens)
print(f"  ✓ Unique tokens in vocabulary: {len(vocabulary):,}")

# Most common tokens
from collections import Counter
all_tokens_flat = [token for tokens in all_movies['tokens'] for token in tokens]
token_freq = Counter(all_tokens_flat)
print(f"\nTop 20 most frequent tokens:")
for token, count in token_freq.most_common(20):
    print(f"  {token:20s} : {count:6,} occurrences")

# Show sample
print("\n[5/5] Sample tokenized documents:")
print("-"*80)
for i in range(3):
    movie = all_movies.iloc[i]
    print(f"\n{i+1}. {movie['title']} ({movie['decade']})")
    print(f"   Original plot length: {len(movie['plot'])} chars")
    print(f"   Tokens ({len(movie['tokens'])}): {movie['tokens'][:15]}...")


print("\nBuilding LSI index")
lsi = LSI(k=200)

token_lists = all_movies['tokens'].tolist()

lsi.fit(token_lists)

print(" ✓ LSI model built successfully!")    