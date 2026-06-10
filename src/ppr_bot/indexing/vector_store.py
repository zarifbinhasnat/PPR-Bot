"""Step 6b: a tiny, transparent vector store built on NumPy.

A "vector store" holds the embedding of every chunk and, given a query
vector, returns the most similar chunks. Production systems use dedicated
databases (Chroma, FAISS, pgvector) with approximate-nearest-neighbour
indexes for million-scale corpora.

We have ~800-1200 chunks. At that scale a brute-force search — compute the
cosine similarity of the query against EVERY chunk with one matrix multiply,
then take the top-k — runs in single-digit milliseconds. So we implement it
by hand in ~30 lines. This is deliberately transparent for teaching: you can
see exactly what "vector similarity search" is, with no black box. (We chose
this over ChromaDB partly for clarity and partly to dodge a known ChromaDB
persistence bug on Windows + Python 3.11.)

Because embeddings are L2-normalized (see embedder.py), cosine similarity is
just the dot product: `scores = embeddings @ query`.
"""

import numpy as np


class NumpyVectorStore:
    """Holds an (n, dim) embedding matrix and the parallel chunk ids."""

    def __init__(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        assert embeddings.shape[0] == len(chunk_ids)
        self.embeddings = embeddings  # (n, dim), unit-normalized rows
        self.chunk_ids = chunk_ids

    @classmethod
    def load(cls, embeddings_path, chunk_ids: list[str]) -> "NumpyVectorStore":
        """Load embeddings from a .npy file, pairing with chunk_ids in order."""
        embeddings = np.load(embeddings_path)
        return cls(embeddings, chunk_ids)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return the top_k (chunk_id, cosine_similarity) for a query vector.

        Steps:
          1. scores = matrix @ query  -> one similarity per chunk
          2. argpartition to grab the top_k indices cheaply (O(n))
          3. sort just those k by score, descending
        """
        scores = self.embeddings @ query_vector  # (n,)
        k = min(top_k, len(self.chunk_ids))
        # argpartition puts the k largest (unordered) in the last k slots.
        top_idx = np.argpartition(scores, -k)[-k:]
        # Now order those k by score, highest first.
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx]
