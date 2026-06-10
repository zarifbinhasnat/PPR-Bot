"""Step 4b: turn parsed `Section`s into retrieval chunks.

Why not just split the document into fixed 512-token windows? Because this
is a *structured legal document*. A user asking "what is the threshold for
direct procurement?" wants the answer scoped to a specific Rule, with that
Rule's number and parent Chapter intact. Fixed-size windows would slice
mid-rule and strip that structure.

So we chunk *along the document's own hierarchy*:
  - Each leaf `Section` (typically a Rule or Schedule sub-section) is the
    natural chunk unit.
  - If a section is too LONG (> CHUNK_MAX_TOKENS), split it into overlapping
    sub-chunks on paragraph/sentence boundaries.
  - Every chunk keeps a `parent_path` breadcrumb and `source_pages` in its
    metadata, so retrieval can show "Rule 42, Part II" and cite the page.

Token counting here is approximate (word/char heuristic) — we don't need a
real tokenizer for chunk-sizing decisions, and avoiding one keeps this step
dependency-free and fast. The embedding model does its own real tokenization
later.
"""

import re
import uuid
from dataclasses import dataclass, field

from ppr_bot.chunking.markdown_parser import Section
from ppr_bot.config import settings


@dataclass
class Chunk:
    chunk_id: str
    text: str  # the raw chunk text
    metadata: dict = field(default_factory=dict)


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens/word is typical for mixed scripts."""
    return int(len(text.split()) * 1.3) + 1


def _split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split over'long text into overlapping windows on paragraph/sentence breaks.

    We first try paragraph boundaries (blank lines); if a single paragraph is
    still too big, we fall back to sentence-ish boundaries (Bangla '।' danda
    and Latin '.?!'). Overlap carries a little context across the cut so a
    fact split across the boundary is still retrievable from either side.
    """
    # Candidate atomic units: paragraphs, further broken by sentence enders.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if _approx_tokens(para) <= max_tokens:
            units.append(para)
        else:
            # Split on Bangla danda or Latin sentence enders, keeping them.
            sentences = re.split(r"(?<=[।.?!])\s+", para)
            units.extend(s.strip() for s in sentences if s.strip())

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = _approx_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            # Start the next window with a tail overlap of the previous one.
            overlap: list[str] = []
            overlap_count = 0
            for prev in reversed(current):
                overlap.insert(0, prev)
                overlap_count += _approx_tokens(prev)
                if overlap_count >= overlap_tokens:
                    break
            current = overlap
            current_tokens = sum(_approx_tokens(u) for u in current)
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_sections(
    sections: list[Section],
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    min_tokens: int = 20,
) -> list[Chunk]:
    """Convert parsed sections into retrieval `Chunk`s with metadata.

    Sections with no body text (pure container headings like a bare Part
    title) are skipped — they carry no answerable content on their own and
    their title still lives in every child's `parent_path`.
    """
    max_tokens = max_tokens or settings.CHUNK_MAX_TOKENS
    overlap_tokens = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS

    chunks: list[Chunk] = []
    for section in sections:
        body = section.text.strip()
        if not body or _approx_tokens(body) < min_tokens:
            continue

        breadcrumb = " > ".join(section.path)
        pages = (
            [section.start_page]
            if section.start_page == section.end_page
            else list(range(section.start_page, section.end_page + 1))
        )

        if _approx_tokens(body) <= max_tokens:
            pieces = [body]
        else:
            pieces = _split_long_text(body, max_tokens, overlap_tokens)

        for i, piece in enumerate(pieces):
            # Prepend the breadcrumb so the chunk is self-describing even in
            # isolation — helps both the embedder and the LLM at answer time.
            chunk_text = f"[{breadcrumb}]\n{piece}"
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata={
                        "title": section.title,
                        "parent_path": section.path,
                        "breadcrumb": breadcrumb,
                        "level": section.level,
                        "source_pages": pages,
                        "part_index": i,  # 0 unless the section was split
                        "part_count": len(pieces),
                    },
                )
            )

    return chunks
