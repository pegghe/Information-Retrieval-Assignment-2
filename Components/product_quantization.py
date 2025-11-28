import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

class ProductQuantizer:
    def __init__(self, n_subvectors=4, n_clusters=256):
        """
        Product Quantization implementation.
        
        Args:
            n_subvectors (m): Number of subspaces to split the vector into.
            n_clusters (k): Number of centroids per subspace (usually 256 to fit in 1 byte).
        """
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.kmeans_models = [] # Will store 'm' KMeans models
        self.codebook = None    # To store the encoded dataset (N x m) integers
        self.d_sub = 0          # Dimension of each subvector

    def fit(self, X):
        """
        Train the quantizer.
        1. Split vectors into m subvectors.
        2. Train a separate KMeans on each subspace.
        """
        n_samples, n_features = X.shape
        assert n_features % self.n_subvectors == 0, "Dimension must be divisible by n_subvectors"
        self.d_sub = n_features // self.n_subvectors
        
        print(f"Training PQ: {n_samples} vectors, {n_features} dims -> {self.n_subvectors} subspaces of dim {self.d_sub}")

        self.kmeans_models = []
        
        # Iterate over each subspace
        for m in range(self.n_subvectors):
            # Extract the subvectors for the m-th slice
            start_col = m * self.d_sub
            end_col = (m + 1) * self.d_sub
            X_sub = X[:, start_col:end_col]
            
            # Train KMeans on this subspace
            # n_init='auto' is faster; usually k=256
            kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=42)
            kmeans.fit(X_sub)
            self.kmeans_models.append(kmeans)
            
        print("Training complete.")

    def encode(self, X):
        """
        Compress the dataset X into codes.
        Returns: matrix (N, m) of integers (cluster IDs).
        NO MORE FLOATS stored for the dataset!
        """
        n_samples = X.shape[0]
        # This will store our compressed representation
        codes = np.zeros((n_samples, self.n_subvectors), dtype=np.uint16) # uint8 if k<=256
        
        for m in range(self.n_subvectors):
            start_col = m * self.d_sub
            end_col = (m + 1) * self.d_sub
            X_sub = X[:, start_col:end_col]
            
            # Predict the nearest cluster ID for each subvector
            model = self.kmeans_models[m]
            codes[:, m] = model.predict(X_sub)
            
        self.codebook = codes
        return codes

    def search(self, query_vec, top_k=10):
        """
        Asymmetric Distance Computation (ADC).
        1. Query is NOT quantized (kept as float).
        2. Dataset is quantized (integers).
        3. Use Lookup Table (LUT) for speed.
        """
        if self.codebook is None:
            raise ValueError("Index not populated. Call encode() first.")

        # 1. Precompute Distance Table (LUT)
        # We calculate distance from Query parts to ALL centroids of that part.
        # Table size: (n_subvectors, n_clusters)
        
        # Reshape query to match subspaces
        # query_vec: (D,)
        d_table = np.zeros((self.n_subvectors, self.n_clusters))
        
        for m in range(self.n_subvectors):
            start_col = m * self.d_sub
            end_col = (m + 1) * self.d_sub
            query_sub = query_vec[start_col:end_col].reshape(1, -1)
            
            # Get centroids for this subspace
            centroids = self.kmeans_models[m].cluster_centers_
            
            # Compute Squared Euclidean distances from query_sub to all 256 centroids
            # We use squared because sqrt is monotonic (doesn't change order)
            dists = euclidean_distances(query_sub, centroids, squared=True).flatten()
            d_table[m, :] = dists

        # 2. Compute approximate distance for all documents using the LUT
        # This is the fast part: just summing up values from the table
        
        # We need to sum the distances for each document based on their codes
        # N documents. For each doc, we look at its m codes, and fetch the partial dists.
        
        n_docs = self.codebook.shape[0]
        doc_distances = np.zeros(n_docs)
        
        # Vectorized lookup is possible but let's do explicit loop for clarity of logic
        # Or better: accumulate per subspace
        for m in range(self.n_subvectors):
            # The column of codes for the m-th subspace
            col_codes = self.codebook[:, m] 
            # Add the corresponding precomputed distance
            doc_distances += d_table[m, col_codes]

        # 3. Sort and return top-k
        nearest_indices = np.argsort(doc_distances)[:top_k]
        nearest_distances = doc_distances[nearest_indices]
        
        return nearest_indices, nearest_distances
