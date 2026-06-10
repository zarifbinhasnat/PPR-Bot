# 05 · Reranking with a cross-encoder

## Bi-encoder vs. cross-encoder

The embedder (`bge-m3`) is a **bi-encoder**: it encodes the query and each
chunk *independently* into vectors, then compares the vectors. This is cheap
and lets you pre-compute every chunk's vector once — so you can search
thousands of chunks fast. But the model never actually looks at the query and
a chunk *together*, so its top-ranking is approximate.

A **cross-encoder** (`bge-reranker-v2-m3`) feeds the query AND a chunk into
the model *together* and outputs a single relevance score. It can attend
across both texts, so it's far more accurate — but far slower, and you can't
pre-compute anything (the score depends on the specific query).

## The two-stage pattern

```
retrieve cheap (bi-encoder + BM25, fused) → top ~20-30 candidates
    → rerank precise (cross-encoder)        → final top 5
```

You get the bi-encoder's speed over the whole corpus AND the cross-encoder's
accuracy at the top, by only running the expensive model on a short list.
This is standard in modern production RAG.

## In this codebase

- `retrieval/reranker.py` — `Reranker.rerank(query, candidates, top_k)` scores
  each `(query, chunk_text)` pair and returns the best `top_k`, annotating
  each with a `rerank_score`.
- `retrieval/pipeline.py` — calls hybrid search first, hydrates the fused ids
  into chunk records, then reranks them.

`bge-reranker-v2-m3` is multilingual (Bangla + English), matching the
embedder, so mixed-script queries rerank correctly.
