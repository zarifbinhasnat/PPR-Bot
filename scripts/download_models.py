"""Pre-download the HuggingFace models (bge-m3 embedder + bge-reranker-v2-m3)
so the first indexing run and server startup don't pay the ~3.3 GB fetch.

Loading each model triggers the download into the local HF cache. We load on
CPU (this project never assumes a GPU) and run a tiny inference to confirm the
weights are usable, not just present on disk.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ppr_bot.config import settings

t0 = time.time()

print(f"[1/2] Downloading embedder: {settings.EMBEDDING_MODEL_NAME} (~2.2 GB)...")
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")
vec = embedder.encode(["পরীক্ষা / test"], normalize_embeddings=True)
print(f"      embedder OK — dim={vec.shape[1]} ({time.time() - t0:.0f}s elapsed)")

print(f"[2/2] Downloading reranker: {settings.RERANKER_MODEL_NAME} (~1.1 GB)...")
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(settings.RERANKER_MODEL_NAME, device="cpu")
score = reranker.predict([("test query", "test document")])
print(f"      reranker OK — score={float(score[0]):.3f}")

print(f"\nDone. Both models cached. Total {time.time() - t0:.0f}s.")
