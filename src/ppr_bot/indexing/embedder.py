"""Step 6a: the dense embedding model wrapper (BAAI/bge-m3).

An "embedding" turns a piece of text into a fixed-length vector of numbers
such that texts with similar *meaning* land near each other in vector space.
That's what enables *semantic* search: a query about "direct purchase limits"
can match a chunk that says "সরাসরি ক্রয়ের সীমা" even with no shared words.

We use BAAI/bge-m3 because it's strongly multilingual (handles Bangla AND
English in one shared space, so a Bangla query can even match English text),
free, and runs locally on CPU. It outputs 1024-dimensional vectors.

The model (~2.2GB) is downloaded automatically on first use and cached.
"""

from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer

from ppr_bot.config import settings


class Embedder:
    """Lazy-loading wrapper around the bge-m3 sentence-transformer.

    Loading the model is expensive (seconds + GBs of RAM), so we load it
    once and reuse it. In the API server it's loaded a single time at
    startup (see api/main.py) — never per request.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME

    @cached_property
    def _model(self) -> SentenceTransformer:
        # device="cpu" because we install the CPU build of torch.
        return SentenceTransformer(self.model_name, device="cpu")

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed a list of texts -> (n, dim) float32 array.

        normalize_embeddings=True scales each vector to unit length, so a
        plain dot product between two vectors equals their cosine similarity
        — which is what our vector store uses for ranking.
        """
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string -> (dim,) float32 vector."""
        return self.embed_texts([query], batch_size=1)[0]
