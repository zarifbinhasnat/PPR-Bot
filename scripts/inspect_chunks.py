"""Quick look at chunks.jsonl quality: print a few chunk records (metadata +
text preview) to sanity-check chunking before running enrichment/indexing."""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ppr_bot.config import settings

with open(settings.chunks_path, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

print("total chunks:", len(chunks))
indices = [0, 1, 2, len(chunks) // 2, len(chunks) - 1]
for i in indices:
    c = chunks[i]
    print("=" * 60)
    print(f"[{i}] id: {c['chunk_id']}")
    print("meta:", c["metadata"])
    print("text:")
    print(c["text"][:400])
    if c.get("contextual_summary"):
        print("--- contextual_summary ---")
        print(c["contextual_summary"])
