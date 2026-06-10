"""Step 7a: hybrid search = dense + sparse, fused with Reciprocal Rank Fusion.

We have two retrievers that return results on totally different scales:
  - dense (vector store): cosine similarities, ~0.0 to 1.0
  - sparse (BM25):        unbounded positive scores, e.g. 3.7, 12.1

You can't just add those scores — the scales are incomparable. Reciprocal
Rank Fusion (RRF) sidesteps this elegantly: it ignores the raw scores and
uses only each document's *rank position* in each list. A document's fused
score is the sum, over every result list, of 1 / (k + rank).

  - rank is 1-based (best result = rank 1).
  - k (default 60) is a smoothing constant that limits how much the very top
    ranks dominate. It's the standard value from the original RRF paper.

A document near the top of BOTH lists wins; a document ranked highly by just
one retriever still scores respectably. This simple, parameter-light method
is a remarkably strong baseline and an industry standard.
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked lists of ids into one, by RRF.

    Args:
        ranked_lists: each inner list is chunk_ids ordered best-first.
        k: RRF smoothing constant.

    Returns:
        (chunk_id, fused_score) sorted by score descending.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    embedder,
    vector_store,
    bm25_index,
    top_k_dense: int,
    top_k_sparse: int,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Run dense + sparse retrieval and fuse their rankings with RRF.

    Returns fused (chunk_id, score) best-first. The caller typically takes
    the top ~20-30 of these to hand to the reranker.
    """
    query_vector = embedder.embed_query(query)
    dense_hits = vector_store.search(query_vector, top_k_dense)
    sparse_hits = bm25_index.search(query, top_k_sparse)

    dense_ids = [cid for cid, _ in dense_hits]
    sparse_ids = [cid for cid, _ in sparse_hits]
    return reciprocal_rank_fusion([dense_ids, sparse_ids], k=rrf_k)
