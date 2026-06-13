"""Recheck (a) the Gemini free-tier daily quota across the models we use, and
(b) whether the required HuggingFace models are already downloaded locally."""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ppr_bot.config import settings
from ppr_bot.llm_client import get_client

# ── (a) Gemini quota probe ──────────────────────────────────────────────────
client = get_client()
models = [
    settings.GEMINI_OCR_MODEL,
    "gemini-flash-latest",
    settings.GEMINI_AUX_MODEL,
    "gemini-flash-lite-latest",
]
print("=== Gemini quota probe ===")
for m in dict.fromkeys(models):  # de-dupe, keep order
    try:
        r = client.models.generate_content(model=m, contents="ping")
        ok = bool((r.text or "").strip())
        print(f"  {m:28s} OK (responded)")
    except Exception as exc:
        s = str(exc)
        if "RESOURCE_EXHAUSTED" in s or "429" in s:
            print(f"  {m:28s} QUOTA EXHAUSTED (429)")
        else:
            print(f"  {m:28s} ERROR: {s[:80]}")

# ── (b) HuggingFace model cache check ───────────────────────────────────────
print("\n=== HuggingFace model cache ===")
import os
from pathlib import Path

hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
print("cache dir:", hub)
wanted = [settings.EMBEDDING_MODEL_NAME, settings.RERANKER_MODEL_NAME]
for name in wanted:
    folder = "models--" + name.replace("/", "--")
    path = hub / folder
    if path.exists():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"  {name:28s} DOWNLOADED ({size/1e6:.0f} MB) at {path.name}")
    else:
        print(f"  {name:28s} NOT downloaded (will fetch on first indexing run)")
