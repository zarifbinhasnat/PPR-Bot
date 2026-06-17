import sys, unicodedata, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

# How many chunks are affected by NFC normalization (i.e. text changes shape)?
chunks_path = Path("C:/Users/Zarif/Documents/PPR-Bot/data/processed/chunks.jsonl")
chunks = [json.loads(l) for l in chunks_path.open(encoding="utf-8") if l.strip()]

affected = 0
for c in chunks:
    t = c.get("indexed_text") or c["text"]
    if unicodedata.normalize("NFC", t) != t:
        affected += 1
print(f"Chunks whose text changes under NFC: {affected} / {len(chunks)}")

# Count the specific decomposed sequences that NFC would compose
pairs = {
    "য় (YA+NUKTA -> YYA)": ("য়", "য়"),
    "ড় (DDA+NUKTA -> RRA)": ("ড়", "ঢ়"),
    "ঢ় (DHA+NUKTA -> RHA)": ("ঢ়", "৞"),
}
full = "\n".join((c.get("indexed_text") or c["text"]) for c in chunks)
for label, (decomp, comp) in pairs.items():
    print(f"  {label}: decomposed={full.count(decomp)}  precomposed={full.count(comp)}")
