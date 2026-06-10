"""Step 4 (orchestration): full_document.md -> chunks.jsonl.

Reads the concatenated Markdown produced by extraction, parses it into a
hierarchy (`markdown_parser`), chunks it (`chunker`), and writes one JSON
object per line to `data/processed/chunks.jsonl`.

JSONL (one JSON per line) is the standard format for this kind of record
stream: append-friendly, streamable, and trivially inspectable with tools.

Usage (from project root):
    python -m ppr_bot.chunking.run_chunking
"""

import json
import sys

from ppr_bot.chunking.chunker import chunk_sections
from ppr_bot.chunking.markdown_parser import parse_markdown
from ppr_bot.config import settings

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    if not settings.full_document_path.exists():
        raise SystemExit(
            f"{settings.full_document_path} not found. Run extraction first."
        )

    markdown = settings.full_document_path.read_text(encoding="utf-8")
    sections = parse_markdown(markdown)
    chunks = chunk_sections(sections)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with settings.chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Parsed {len(sections)} sections -> {len(chunks)} chunks")
    print(f"Wrote {settings.chunks_path}")


if __name__ == "__main__":
    main()
