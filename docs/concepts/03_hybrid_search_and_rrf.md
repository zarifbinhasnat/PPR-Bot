# 03 · Hybrid search & Reciprocal Rank Fusion

## Two kinds of search, different strengths

**Dense (vector) search** — `indexing/embedder.py` + `vector_store.py`.
Embeddings put text with similar *meaning* near each other. A query about
"direct purchase limits" can match "সরাসরি ক্রয়ের সীমা" with no shared words.
Weakness: it can miss *exact* tokens — a specific rule number, an acronym
like "BPPA", a rare term.

**Sparse (keyword/BM25) search** — `indexing/bm25_index.py`.
BM25 scores documents by term overlap with the query (improved TF-IDF).
Great at exact matches; blind to meaning/synonyms.

**Hybrid search** runs both and combines them. It reliably beats either alone,
because most queries need a bit of both.

## The fusion problem

Dense scores are cosine similarities (~0–1). BM25 scores are unbounded (3.7,
12.1, …). You **cannot just add them** — the scales are incomparable, and
naïve normalization is fragile.

## Reciprocal Rank Fusion (RRF)

RRF ignores the raw scores and uses only each document's **rank position** in
each list:

```
score(d) = Σ over each result list of   1 / (k + rank_in_that_list(d))
```

- `rank` is 1-based (best = 1).
- `k` (default 60, from the original paper) damps how much the very top ranks
  dominate.

A doc near the top of *both* lists wins; a doc only one retriever found still
scores decently. It's tiny, parameter-light, and a genuine industry standard.

Implementation: `retrieval/hybrid_search.py` (`reciprocal_rank_fusion`), with
hand-computed expectations in `tests/test_rrf.py`.

```python
fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=60)
```
