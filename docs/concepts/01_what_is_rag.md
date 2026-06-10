# 01 · What is RAG?

**RAG = Retrieval-Augmented Generation.** Instead of asking a language model
to answer from its own (fixed, possibly outdated, possibly hallucinated)
memory, you:

1. **Retrieve** the most relevant passages from *your* documents, then
2. **Generate** an answer that is *grounded in those passages*.

## Why not just fine-tune or prompt the model with the whole document?

- **The whole document doesn't fit / is wasteful.** PPR 2025 is ~270 pages.
  Stuffing all of it into every prompt is slow and expensive, and models
  attend poorly to needles in huge haystacks.
- **Fine-tuning bakes in facts you can't easily update or cite.** RAG keeps
  the knowledge in an index you can rebuild any time, and every answer can
  point back to a source page.
- **Grounding fights hallucination.** If the model is told "answer only from
  these excerpts, and say so if they don't cover it," it makes things up far
  less.

## The two phases (mirrored in this codebase)

- **Indexing (offline):** documents → chunks → embeddings + keyword index.
  Done once. See the `ingestion/`, `chunking/`, `enrichment/`, `indexing/`
  packages.
- **Querying (online):** question → retrieve relevant chunks → generate a
  grounded answer. Done per request. See `retrieval/`, `generation/`,
  `chat/`, `api/`.

## Where the "SOTA" lives

A basic RAG does: embed chunks → cosine-search top-k → stuff into prompt.
This project layers on the techniques that separate a toy from a robust
system:

- **Hybrid search** (dense + keyword) — `concepts/03`
- **Contextual Retrieval** — `concepts/04`
- **Reranking** — `concepts/05`
- **Query transformation** — `concepts/06`

Each is explained in its own note.
