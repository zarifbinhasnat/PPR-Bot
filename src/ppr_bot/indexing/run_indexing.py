"""Step 6 (orchestration): build both indexes from chunks.jsonl.

Reads enriched chunks, embeds their `indexed_text` (context blurb + chunk)
with bge-m3, and writes:
  - data/processed/embeddings.npy  (the dense vector matrix)
  - data/processed/bm25_index.pkl  (the sparse keyword index)

The chunk order in chunks.jsonl defines the row order of embeddings.npy and
the id order of the BM25 index — they must stay aligned, so we read the
chunk ids once and reuse them for both.

Usage (from project root):
    python -m ppr_bot.indexing.run_indexing
"""

import json
import sys
import time

import numpy as np

from ppr_bot.config import settings
from ppr_bot.indexing.bm25_index import BM25Index
from ppr_bot.indexing.embedder import Embedder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    if not settings.chunks_path.exists():
        raise SystemExit(f"{settings.chunks_path} not found. Run chunking first.")

    chunks: list[dict] = []
    with settings.chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    chunk_ids = [c["chunk_id"] for c in chunks]
    # Prefer the enriched indexed_text; fall back to raw text if enrichment
    # hasn't been run yet.
    corpus = [c.get("indexed_text") or c["text"] for c in chunks]
    print(f"Indexing {len(chunks)} chunks...")

    # --- Dense index ---
    t0 = time.time()
    embedder = Embedder()
    embeddings = embedder.embed_texts(corpus)
    np.save(settings.embeddings_path, embeddings)
    print(f"Embeddings {embeddings.shape} saved in {time.time() - t0:.1f}s")

    # --- Sparse index ---
    bm25 = BM25Index.build(corpus, chunk_ids)
    bm25.save(settings.bm25_index_path)
    print(f"BM25 index saved ({len(chunk_ids)} docs)")
    print(f"Wrote {settings.embeddings_path} and {settings.bm25_index_path}")


if __name__ == "__main__":
    main()
