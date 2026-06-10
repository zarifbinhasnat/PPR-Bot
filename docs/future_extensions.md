# Future extensions (v2 ideas)

These are deliberately out of scope for v1 to keep the core teachable, but
each is a natural next step.

## 1. GraphRAG — a cross-reference graph over the Rules

Legal text is full of internal references: "subject to Rule 48", "as per
Schedule 3", "notwithstanding sub-rule (2)". Flat vector search can retrieve
the chunk you asked about but miss the rule it *depends on*.

**Sketch for PPR-Bot:**
- During chunking, extract reference edges with a regex/LLM pass:
  `Rule 76 --references--> Schedule 3`, `Rule 76 --references--> Rule 48`.
- Store as a small graph (e.g. `networkx`, or a `references` field per chunk).
- At retrieval time, after the normal hybrid+rerank step, **expand** the
  result set by pulling in chunks that the top hits reference (1 hop). This
  gives the generator the referenced rules even if they didn't match the query
  lexically/semantically — enabling multi-hop legal reasoning ("what's the
  threshold, *and* what approval does that trigger?").
- This is a focused, document-specific GraphRAG that's far simpler than a
  general entity-relation knowledge graph, yet captures most of the value here.

## 2. Durable conversation memory (SQLite)

`chat/memory.py` already defines the `ConversationStore` interface. Add a
`SQLiteConversationStore`: a `turns(session_id, idx, role, content, ts)` table,
INSERT on `append_turn`, SELECT ordered by `idx` on `get_history`. Nothing
else changes — that's the payoff of programming to the interface.

## 3. Evaluation harness

Right now quality is judged by eyeballing. Add:
- A small labelled QA set (question → expected rule/page).
- **Retrieval metrics**: recall@k, MRR — does the right chunk appear in the
  top-k?
- **Answer metrics**: an LLM-as-judge scoring groundedness/citation accuracy.
- Wire it so config changes (chunk size, `TOP_K_*`, with/without enrichment)
  can be compared quantitatively instead of by vibes.

## 4. Scaling the vector store

The NumPy brute-force store is perfect at ~1k chunks. For 100k+ swap in
`faiss-cpu` (HNSW index) behind the same `search()` signature, or ChromaDB /
pgvector if you need a managed DB. The `NumpyVectorStore` interface is small
on purpose.

## 5. Query expansion / HyDE

Add `expand_query()` (paraphrases, fused via RRF) and/or HyDE (embed a
hypothetical answer). Both trade extra LLM calls for recall. See
`concepts/06`.

## 6. Better Bangla tokenization for BM25

`simple_tokenize` is script-aware but does no stemming/normalization. If
Bangla keyword recall proves weak, integrate a Bangla NLP tokenizer
(e.g. `bnlp`) behind `simple_tokenize`.
