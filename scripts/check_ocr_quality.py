"""Scan the extracted per-page Markdown for OCR quality problems.

We OCR the *rendered image* (not the PDF text layer), so the classic
SutonnyMJ mojibake shouldn't appear — but this verifies that, and also
catches the other failure modes we've hit: empty/near-empty output, and
degenerate single-character repetition loops.

Per page it computes:
  - size in bytes
  - Bangla character ratio (Unicode ঀ–৿)
  - "suspicious symbol" ratio — Latin-1 supplement letters + dagger/bullet
    glyphs (‡ † ´ ` Ò Ó …) that are the hallmark of legacy-font mojibake but
    are rare in clean Bangla/English
  - longest run of a single repeated character

A page is FLAGGED if it's near-empty, has a high mojibake-symbol ratio, or
contains a long repetition run. Legitimately-English pages (RFP templates,
flowcharts) have low Bangla ratio but clean ASCII + near-zero suspicious
symbols, so they are NOT flagged.

Usage:
    venv/Scripts/python.exe scripts/check_ocr_quality.py
"""

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ppr_bot.config import settings

BANGLA = re.compile(r"[ঀ-৿]")
# Latin-1 supplement letters + the specific symbol glyphs SutonnyMJ mojibake
# produces. Clean Bangla/English almost never uses these.
SUSPICIOUS = re.compile(r"[À-ÿ†‡•‘’“”´`ˆ˜]")
# A real degenerate loop is the same LETTER/glyph repeated many times. Long
# runs of whitespace (header padding) or structural punctuation — Markdown
# table separators "|:------|", leader dots, underscores, horizontal rules —
# are legitimate, so we exclude those characters from the check.
REPEAT_RUN = re.compile(r"([^\s\-=|:._*#~/\\+])\1{29,}")

NEAR_EMPTY_BYTES = 40
SUSPICIOUS_RATIO_FLAG = 0.04  # >4% weird symbols = likely mojibake


def analyze(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    # Ignore the gazette_page comment + whitespace for ratio purposes.
    body = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
    n = len(body) or 1
    bangla = len(BANGLA.findall(body))
    suspicious = len(SUSPICIOUS.findall(body))
    run = REPEAT_RUN.search(body)
    return {
        "bytes": len(text.encode("utf-8")),
        "chars": len(body),
        "bangla_ratio": bangla / n,
        "suspicious_ratio": suspicious / n,
        "repeat_run": bool(run),
        "is_placeholder": "no transcribable content" in text,
    }


def main() -> None:
    files = sorted(settings.pages_markdown_dir.glob("page_*.md"))
    if not files:
        raise SystemExit("No extracted pages found yet.")

    flagged: list[tuple[str, str]] = []
    ok = 0
    for f in files:
        a = analyze(f)
        reasons = []
        if a["is_placeholder"]:
            ok += 1  # explicit "no content" placeholder — fine
            continue
        if a["bytes"] < NEAR_EMPTY_BYTES:
            reasons.append(f"near-empty ({a['bytes']}B)")
        if a["suspicious_ratio"] > SUSPICIOUS_RATIO_FLAG:
            reasons.append(f"mojibake? {a['suspicious_ratio']:.0%} odd symbols")
        if a["repeat_run"]:
            reasons.append("repetition run")
        if reasons:
            flagged.append((f.name, "; ".join(reasons)))
        else:
            ok += 1

    print(f"Checked {len(files)} pages: {ok} clean, {len(flagged)} flagged.\n")
    if flagged:
        print("FLAGGED PAGES:")
        for name, why in flagged:
            print(f"  {name}: {why}")
    else:
        print("No gibberish/empty/repetition issues detected. ✔")

    # Quick distribution sanity line.
    ratios = [analyze(f)["bangla_ratio"] for f in files]
    avg = sum(ratios) / len(ratios)
    print(f"\nAvg Bangla character ratio across pages: {avg:.0%} "
          f"(low-Bangla pages are usually legit English templates).")


if __name__ == "__main__":
    main()
