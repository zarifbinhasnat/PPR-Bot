# PPR-Bot Architecture

This document explains how the whole system fits together. Read it alongside
the code — every module named here has heavy inline comments.

## Two halves: offline pipeline vs. online serving

The system splits cleanly into work done **once, ahead of time** (turning a
messy PDF into searchable indexes) and work done **per user message** (answer
a question). This separation is the most important structural idea in the
project.

```
                 OFFLINE PIPELINE (run once, produces artifacts)
 ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
 │  PDF     │ → │ render   │ → │ Gemini   │ → │ structure- │ → │contextual│
 │ 271 pgs  │   │ to PNGs  │   │ OCR      │   │ aware      │   │enrichment│
 │          │   │ (PyMuPDF)│   │ (vision) │   │ chunking   │   │ (LLM)    │
 └──────────┘   └──────────┘   └────┬─────┘   └─────┬──────┘   └────┬─────┘
                                    │ page md        │ chunks.jsonl  │
                                    ▼                ▼               ▼
                              full_document.md   chunks.jsonl   chunks.jsonl
                                                               (+ indexed_text)
                                                                    │
                                                       ┌────────────┴───────────┐
                                                       ▼                        ▼
                                                  embeddings.npy           bm25_index.pkl
                                                  (bge-m3 dense)           (sparse/keyword)

                 ONLINE SERVING (per request)
 user msg ─► condense query ─► hybrid search ─► RRF fuse ─► rerank ─► generate
            (LLM, uses        (dense + sparse)            (cross-     (Gemini,
             chat history)                                 encoder)    streamed)
                                                                          │
                                                                          ▼
                                                                    SSE → browser
```

## Offline pipeline stages

| Stage | Module | Input → Output | Teaches |
|---|---|---|---|
| 1 Render | `ingestion/pdf_renderer.py` | PDF → page PNGs | PyMuPDF rasterization |
| 2 OCR | `ingestion/ocr_transcriber.py` | PNG → Markdown | multimodal LLM prompting, retry/fallback |
| 3 Orchestrate | `ingestion/run_extraction.py` | all pages → `full_document.md` | resumable batch jobs, manifests |
| 4 Chunk | `chunking/` | Markdown → `chunks.jsonl` | structure-aware chunking |
| 5 Enrich | `enrichment/` | chunks → `+contextual_summary` | Contextual Retrieval |
| 6 Index | `indexing/` | chunks → `embeddings.npy` + `bm25_index.pkl` | embeddings, vector search, BM25 |

## Online serving stages

| Stage | Module | Teaches |
|---|---|---|
| Condense | `retrieval/query_transform.py` | conversational query rewriting |
| Hybrid + RRF | `retrieval/hybrid_search.py` | reciprocal rank fusion |
| Rerank | `retrieval/reranker.py` | cross-encoder reranking |
| Pipeline | `retrieval/pipeline.py` | retrieve→rerank orchestration |
| Generate | `generation/answer_generator.py` | grounded, citing, streamed generation |
| Memory | `chat/memory.py` | session state behind an interface |
| Orchestrate | `chat/orchestrator.py` | tying RAG components together |
| API | `api/` | FastAPI lifespan, routers, SSE |

## Key design decisions

- **Vision OCR for everything** — the PDF's legacy SutonnyMJ font makes direct
  text extraction produce mojibake, and ~30 pages are image-only scans. One
  vision-OCR path handles both uniformly.
- **NumPy vector store, not ChromaDB** — at ~1k chunks brute-force cosine
  search is instant and fully transparent; also avoids a known ChromaDB
  Windows/Py3.11 persistence bug.
- **Hand-written everything (no LangChain)** — so each RAG concept is visible.
- **Models loaded once at startup** — see `api/main.py` lifespan; never
  per-request.

See `docs/concepts/` for a deeper explainer on each technique, and
`docs/future_extensions.md` for what a v2 would add (GraphRAG, durable memory,
evaluation harness).
