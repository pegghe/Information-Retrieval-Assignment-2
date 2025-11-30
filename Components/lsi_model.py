import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from Tokenizer import tokenize


class LSI:
    """
    Latent Semantic Indexing (LSI), following Manning et al.
    Chapter 18, 'Latent semantic indexing'.

    Main formulas:
        C = U Σ V^T
        D_k = Σ_k (V_k)^T             (documents in latent space)
        q_k = Σ_k^{-1} U_k^T q       (query projection)
    """

    def __init__(self, k=200):
        """
        :param k: number of latent dimensions to keep (rank-k approximation)
        """
        self.k = k
        self.vocab = None
        self.idf = None
        self.C = None
        self.Uk = None
        self.Sk = None
        self.VkT = None
        self.doc_vectors = None
        self.documents = None

    # -------------------------------------------------------
    # Build vocabulary
    # -------------------------------------------------------
    def _build_vocab(self, token_lists):
        vocab = {}
        for tokens in token_lists:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        return vocab

    # -------------------------------------------------------
    # Build term-document matrix C (TF-IDF weighted)
    # -------------------------------------------------------
    def _build_matrix(self, token_lists):
        """
        Creates TF-IDF matrix C, where:
            C[i, j] = tf(i,j) * idf(i)
        using the IDF variant consistent with Manning.

        :return: C matrix (dense), idf vector
        """
        vocab = self.vocab
        V = len(vocab)
        N = len(token_lists)

        tf = np.zeros((V, N))
        df = np.zeros(V)

        # Term Frequency (TF)
        for j, tokens in enumerate(token_lists):
            counts = {}
            for t in tokens:
                idx = vocab[t]
                counts[idx] = counts.get(idx, 0) + 1

            for idx, count in counts.items():
                tf[idx, j] = count
                df[idx] += 1

        # IDF: log((N+1)/(df+1)) + 1  (Manning's smoothing)
        idf = np.log((N + 1) / (df + 1)) + 1

        # TF-IDF
        C = tf * idf[:, None]

        return C, idf

    # -------------------------------------------------------
    # Fit LSI model
    # -------------------------------------------------------
    def fit(self, token_lists):
        """
        :param token_lists: list of token lists for each document
        """
        self.documents = token_lists

        # 1. Build vocabulary
        self.vocab = self._build_vocab(token_lists)

        # 2. Build C (TF-IDF matrix)
        self.C, self.idf = self._build_matrix(token_lists)

        # 3. Convert C to sparse format for large-scale SVD
        C_sparse = csr_matrix(self.C)

        # 4. Compute truncated SVD (rank-k)
        #    svds returns singular values not sorted → we sort them
        Uk, Sk, VkT = svds(C_sparse, k=self.k)

        idx = np.argsort(-Sk)  # sort by descending singular value
        self.Sk = Sk[idx]
        self.Uk = Uk[:, idx]
        self.VkT = VkT[idx, :]

        # 5. Documents in LSI space:
        #    D_k = Σ_k (V_k)^T
        self.doc_vectors = self.Sk[:, None] * self.VkT

    # -------------------------------------------------------
    # Transform a query into the LSI space (Eq. 18.22)
    # -------------------------------------------------------
    def transform_query(self, query_tokens):
        """
        Convert a tokenized query into the k-dimensional LSI vector.

        Formula:
            q_k = Σ_k^{-1} U_k^T q
        """
        q = np.zeros(len(self.vocab))

        # Build raw TF vector
        for t in query_tokens:
            if t in self.vocab:
                q[self.vocab[t]] += 1

        # Apply IDF weighting
        q = q * self.idf

        # Project into LSI space
        qk = (1 / self.Sk)[:, None] * (self.Uk.T @ q)
        return qk

    # -------------------------------------------------------
    # Search for top-k similar documents using cosine similarity
    # -------------------------------------------------------
    def search(self, query, top_k=10):
        """
        :param query: string query
        :param top_k: number of results
        :return: (indices, similarity scores)
        """
        query_tokens = tokenize(query)
        qk = self.transform_query(query_tokens)

        # cosine similarity in latent space
        norms_docs = np.linalg.norm(self.doc_vectors, axis=0)
        norm_q = np.linalg.norm(qk)

        scores = (self.doc_vectors.T @ qk) / (norms_docs * norm_q + 1e-10)

        ranking = np.argsort(-scores)
        return ranking[:top_k], scores[ranking[:top_k]]