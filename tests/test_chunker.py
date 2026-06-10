"""Unit tests for markdown parsing + chunking."""

from ppr_bot.chunking.chunker import chunk_sections
from ppr_bot.chunking.markdown_parser import parse_markdown

SAMPLE = """<!-- page: 1 -->
# প্রথম অংশ
## প্রথম অধ্যায়
### ১। সংক্ষিপ্ত শিরোনাম।—
(১) এই বিধিমালা পাবলিক প্রকিউরমেন্ট বিধিমালা, ২০২৫ নামে অভিহিত হইবে।
(২) ইহা অবিলম্বে কার্যকর হইবে।
<!-- page: 2 -->
### ২। সংজ্ঞার্থ।—
এই বিধিমালায় বিষয় বা প্রসঙ্গের পরিপন্থী কিছু না থাকিলে।
"""


def test_parse_builds_hierarchy_path():
    sections = parse_markdown(SAMPLE)
    titles = {s.title for s in sections}
    assert "১। সংক্ষিপ্ত শিরোনাম।—" in titles
    rule1 = next(s for s in sections if s.title.startswith("১।"))
    # Its ancestry path should include Part and Chapter.
    assert rule1.path == ["প্রথম অংশ", "প্রথম অধ্যায়", "১। সংক্ষিপ্ত শিরোনাম।—"]


def test_page_markers_tracked():
    sections = parse_markdown(SAMPLE)
    rule2 = next(s for s in sections if s.title.startswith("২।"))
    assert rule2.start_page == 2


def test_chunks_have_breadcrumb_and_metadata():
    sections = parse_markdown(SAMPLE)
    chunks = chunk_sections(sections, min_tokens=1)
    assert chunks, "expected at least one chunk"
    sample = chunks[0]
    assert "breadcrumb" in sample.metadata
    assert "source_pages" in sample.metadata
    # Breadcrumb is prepended to the chunk text.
    assert sample.text.startswith("[")
