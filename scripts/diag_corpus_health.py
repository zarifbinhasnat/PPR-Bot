import sys, json, unicodedata, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

chunks_path = Path("C:/Users/Zarif/Documents/PPR-Bot/data/processed/chunks.jsonl")
full_path = Path("C:/Users/Zarif/Documents/PPR-Bot/data/processed/full_document.md")

chunks = [json.loads(l) for l in chunks_path.open(encoding="utf-8") if l.strip()]
full = full_path.read_text(encoding="utf-8")

print("=== CORPUS STATS ===")
print(f"full_document.md size : {len(full):,} chars")
print(f"Total chunks          : {len(chunks)}")

enriched = sum(1 for c in chunks if c.get("contextual_summary"))
print(f"Enriched chunks       : {enriched}/{len(chunks)}")

empty_text = [c["chunk_id"] for c in chunks if not c.get("text", "").strip()]
print(f"Empty-text chunks     : {len(empty_text)}")
for cid in empty_text:
    print(f"  {cid}")

lengths = sorted(len(c["text"]) for c in chunks)
print(f"Chunk lengths (chars) : min={lengths[0]}  median={lengths[len(lengths)//2]}  max={lengths[-1]}")

very_small = [c["chunk_id"] for c in chunks if len(c.get("text","")) < 50]
print(f"Very small chunks <50 chars: {len(very_small)}")
for cid in very_small[:5]:
    c = next(x for x in chunks if x["chunk_id"] == cid)
    print(f"  {cid}: {c['text'][:80]!r}")

print()
print("=== UNICODE ===")
nfc_bad = sum(1 for c in chunks if unicodedata.normalize("NFC", c.get("indexed_text") or c["text"]) != (c.get("indexed_text") or c["text"]))
print(f"Chunks needing NFC fix: {nfc_bad}/{len(chunks)}")

print()
print("=== ENRICHMENT QUALITY ===")
short = [c["chunk_id"] for c in chunks if c.get("contextual_summary") and len(c["contextual_summary"]) < 30]
print(f"Suspiciously short summaries: {len(short)}")
for cid in short[:5]:
    c = next(x for x in chunks if x["chunk_id"] == cid)
    print(f"  {cid}: {c['contextual_summary']!r}")

print()
print("=== PAGE COVERAGE ===")
page_markers = re.findall(r"<!-- page: (\d+) -->", full)
print(f"Pages in full_document: {len(page_markers)}  ({page_markers[0]}..{page_markers[-1]})")
missing = sorted(set(range(1, 272)) - set(int(p) for p in page_markers))
print(f"Missing pages: {missing if missing else 'none'}")

# Check for very short pages in the raw MD
pages_dir = Path("C:/Users/Zarif/Documents/PPR-Bot/data/pages_markdown")
short_pages = []
for pg in sorted(pages_dir.glob("page_*.md")):
    size = pg.stat().st_size
    if size < 200:
        short_pages.append((pg.name, size))
print(f"\nVery short page files (<200 bytes): {len(short_pages)}")
for name, size in short_pages:
    print(f"  {name}: {size} bytes")
