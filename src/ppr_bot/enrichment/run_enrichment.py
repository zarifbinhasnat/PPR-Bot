"""Step 5 (orchestration): add `contextual_summary` + `indexed_text` to chunks.

Reads `chunks.jsonl`, generates a context blurb for each chunk via Gemini,
and rewrites `chunks.jsonl` in place with two new fields:
  - contextual_summary: the LLM-generated situating blurb
  - indexed_text: blurb + original text (what indexing will embed/BM25)

Resumable: chunks that already have a non-empty `contextual_summary` are
skipped, so an interrupted run can be re-run safely. This matters because
this is ~800-1200 LLM calls.

Usage (from project root):
    python -m ppr_bot.enrichment.run_enrichment
    python -m ppr_bot.enrichment.run_enrichment --limit 20   # test on a sample
"""

import argparse
import json
import sys

from ppr_bot.config import settings
from ppr_bot.enrichment.contextualizer import build_indexed_text, generate_context
from ppr_bot.llm_client import get_client

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_chunks() -> list[dict]:
    with settings.chunks_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_chunks(chunks: list[dict]) -> None:
    with settings.chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual enrichment of chunks.")
    parser.add_argument("--limit", type=int, default=None, help="only first N chunks")
    args = parser.parse_args()

    if not settings.GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY is empty. Set it in .env first.")
    if not settings.chunks_path.exists():
        raise SystemExit(f"{settings.chunks_path} not found. Run chunking first.")

    chunks = _load_chunks()
    client = get_client()

    targets = chunks if args.limit is None else chunks[: args.limit]
    done = 0
    for i, chunk in enumerate(targets):
        if chunk.get("contextual_summary"):
            continue  # already enriched — resume support
        breadcrumb = chunk.get("metadata", {}).get("breadcrumb", "")
        blurb = generate_context(
            chunk["text"], breadcrumb, client, settings.GEMINI_AUX_MODEL
        )
        chunk["contextual_summary"] = blurb
        chunk["indexed_text"] = build_indexed_text(blurb, chunk["text"])
        done += 1
        if done % 20 == 0:
            _save_chunks(chunks)  # periodic checkpoint
            print(f"[checkpoint] enriched {done} chunks (at index {i})")

    _save_chunks(chunks)
    print(f"Enriched {done} chunks. Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
