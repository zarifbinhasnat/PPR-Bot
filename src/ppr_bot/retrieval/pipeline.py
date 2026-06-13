"""Step 7c: the retrieval pipeline that ties the stages together.

This is the "R" in RAG. Given a query string it runs:

    hybrid search (dense + sparse + RRF)   -> ~20-30 candidate ids
        -> look up the candidate chunk records
        -> cross-encoder rerank             -> final top-k chunks

It loads all artifacts produced by the offline pipeline (chunks.jsonl,
embeddings.npy, bm25_index.pkl) and holds the embedder + reranker models. In
the API server one `RetrievalPipeline` is built once at startup and reused
for every request.
"""

import json

from ppr_bot.config import settings
from ppr_bot.indexing.bm25_index import BM25Index
from ppr_bot.indexing.embedder import Embedder
from ppr_bot.indexing.vector_store import NumpyVectorStore
from ppr_bot.retrieval.hybrid_search import hybrid_search
from ppr_bot.retrieval.reranker import Reranker


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        # Load chunk records and build an id -> record lookup.
        self.chunks_by_id: dict[str, dict] = {}
        chunk_ids: list[str] = []
        with settings.chunks_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self.chunks_by_id[record["chunk_id"]] = record
                    chunk_ids.append(record["chunk_id"])

        self.embedder = embedder or Embedder()
        self.reranker = reranker or Reranker()
        self.vector_store = NumpyVectorStore.load(settings.embeddings_path, chunk_ids)
        self.bm25_index = BM25Index.load(settings.bm25_index_path)

    def retrieve(
        self, query: str, top_k: int | None = None, rerank: bool = True
    ) -> list[dict]:
        """Return the most relevant chunk records for a query, best-first.

        When `rerank` is True each record gets a `rerank_score` from the
        cross-encoder. When False we skip the (CPU-expensive) reranker and
        return the hybrid-search/RRF ordering directly — much faster, with
        slightly less precise top-k ordering. The retrieved *content* is the
        same; only the ordering differs.
        """
        top_k = top_k or settings.TOP_K_RERANK

        fused = hybrid_search(
            query,
            self.embedder,
            self.vector_store,
            self.bm25_index,
            top_k_dense=settings.TOP_K_DENSE,
            top_k_sparse=settings.TOP_K_SPARSE,
            rrf_k=settings.RRF_K,
        )

        # Hydrate the fused ids into full chunk records (already RRF-sorted).
        candidates = [
            self.chunks_by_id[cid] for cid, _ in fused if cid in self.chunks_by_id
        ]
        if not rerank:
            return candidates[:top_k]
        return self.reranker.rerank(query, candidates, top_k=top_k)
