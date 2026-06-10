"""Milestone 4 verification: inspect chunking on whatever Markdown exists.

Parses the current full_document.md (even a partial one during extraction),
chunks it, and prints a summary plus the first few chunks with their
metadata so you can eyeball that boundaries align with the Rule/Section
structure and that breadcrumbs/source pages look right.

Usage (from project root):
    python scripts/test_chunking_sample.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppr_bot.chunking.chunker import chunk_sections
from ppr_bot.chunking.markdown_parser import parse_markdown
from ppr_bot.config import settings


def main() -> None:
    if not settings.full_document_path.exists():
        raise SystemExit(f"{settings.full_document_path} not found yet.")

    markdown = settings.full_document_path.read_text(encoding="utf-8")
    sections = parse_markdown(markdown)
    chunks = chunk_sections(sections)

    print(f"Sections parsed: {len(sections)}")
    print(f"Chunks produced: {len(chunks)}\n")

    for chunk in chunks[:6]:
        meta = chunk.metadata
        print("=" * 70)
        print(f"breadcrumb: {meta['breadcrumb']}")
        print(f"pages: {meta['source_pages']}  level: {meta['level']}"
              f"  part {meta['part_index'] + 1}/{meta['part_count']}")
        print("-" * 70)
        preview = chunk.text[:400]
        print(preview + ("..." if len(chunk.text) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
