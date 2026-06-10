# PPR-Bot

A from-scratch, hand-built RAG (Retrieval-Augmented Generation) chatbot over
the Bangladesh Gazette publication of the **Public Procurement Rules (PPR)
2025** — and a teaching codebase for **FastAPI**, **RAG systems**, and
**chat systems**.

No LangChain/LlamaIndex: every step (PDF extraction, chunking, embeddings,
hybrid search, reranking, query rewriting, generation, chat memory, SSE
streaming) is hand-written and commented so you can read the code to learn
how each piece works.

## Why this is a hard document

`PPR 2025.pdf` is a 271-page Bangladesh Gazette. Its text layer is mostly
unusable directly:
- Most body text uses the legacy **SutonnyMJ** Bangla font — extracting it
  with a normal PDF library produces mojibake, not real Bangla Unicode.
- ~30 pages near the end are scanned forms/diagrams with no text layer.
- Some pages are English Standard Tender Document templates.

Our fix: render every page to an image and have a Gemini vision model
transcribe it to clean Markdown — this works uniformly regardless of font or
scan status (see `src/ppr_bot/ingestion/`).

## Setup

1. **Python**: requires Python 3.11+. A venv already exists at `venv/`.

2. **Install dependencies** (CPU-only torch first, to avoid a multi-GB CUDA
   download):
   ```powershell
   venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   venv\Scripts\python.exe -m pip install -r requirements.txt
   venv\Scripts\python.exe -m pip install -e .
   ```

3. **Get a Gemini API key** (free tier available): visit Google AI Studio,
   create an API key, then copy it into `.env`:
   ```
   GEMINI_API_KEY=your-key-here
   ```
   `.env` is already created from `.env.example` with sensible defaults for
   everything else.

## Pipeline (offline, run once)

Each stage reads the previous stage's output and is independently runnable
and resumable:

| Stage | Script | Output |
|---|---|---|
| 1. Render PDF pages to images | `src/ppr_bot/ingestion/pdf_renderer.py` | `data/pages_images/*.png` |
| 2. OCR transcribe (sample) | `scripts/test_extraction_sample.py` | `data/pages_markdown/page_*.md` (sample) |
| 3. Full extraction | `src/ppr_bot/ingestion/run_extraction.py` | `data/pages_markdown/`, `data/processed/full_document.md` |
| 4. Chunking | `src/ppr_bot/chunking/run_chunking.py` | `data/processed/chunks.jsonl` |
| 5. Contextual enrichment | `src/ppr_bot/enrichment/run_enrichment.py` | `chunks.jsonl` (+ `contextual_summary`) |
| 6. Indexing | `src/ppr_bot/indexing/run_indexing.py` | `data/processed/embeddings.npy`, `bm25_index.pkl` |

## Serving (online)

```powershell
venv\Scripts\python.exe -m uvicorn ppr_bot.api.main:app --reload
```

Then open the chat UI at `http://localhost:8000`.

## Architecture

See `docs/architecture.md` and `docs/concepts/` for explanations of each
technique used (hybrid search + RRF, contextual retrieval, reranking, query
transformation, SSE streaming, etc.), written for someone learning these
concepts by reading this code.
