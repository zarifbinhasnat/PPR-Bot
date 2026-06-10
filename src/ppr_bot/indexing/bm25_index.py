"""Step 6c: the sparse (keyword) index using BM25.

Dense embeddings capture *meaning* but can miss *exact* terms — a specific
rule number ("৪২"), an acronym ("BPPA"), or a rare legal term. BM25 is the
classic keyword-ranking algorithm (an improved TF-IDF): it scores a chunk by
how often the query's words appear in it, dampened by how common those words
are across the whole corpus and normalized for chunk length.

Combining BM25 (exact terms) with dense search (meaning) is "hybrid search",
and it reliably beats either one alone. We fuse the two later via RRF.

`rank_bm25` does the scoring but does NOT tokenize for you — and it has no
notion of Bangla. So we provide our own `simple_tokenize` that splits on
whitespace/punctuation and works for both Bangla and Latin scripts. This is a
known simplification (no Bangla stemming/normalization); good enough for v1.
"""

import pickle
import re

from rank_bm25 import BM25Okapi

# Keep runs of letters/digits in any script (incl. Bangla ঀ-৿);
# everything else (spaces, punctuation, dandas) is a separator.
_TOKEN_RE = re.compile(r"[\wঀ-৿]+", re.UNICODE)


def simple_tokenize(text: str) -> list[str]:
    """Lowercase + split into word tokens. Works for Bangla + English."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Wraps BM25Okapi plus the parallel chunk ids and the tokenizer."""

    def __init__(self, bm25: BM25Okapi, chunk_ids: list[str]) -> None:
        self.bm25 = bm25
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, corpus: list[str], chunk_ids: list[str]) -> "BM25Index":
        tokenized = [simple_tokenize(doc) for doc in corpus]
        return cls(BM25Okapi(tokenized), chunk_ids)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, bm25_score) for a query string."""
        scores = self.bm25.get_scores(simple_tokenize(query))
        ranked = sorted(
            zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    @classmethod
    def load(cls, path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["bm25"], data["chunk_ids"])
