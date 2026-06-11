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


# Real OCR output sprinkles repeated Gazette page furniture (printed-page
# comments, running headers, price stamps, issue numbers) right at page
# boundaries — including mid-section, where it would otherwise interrupt a
# chunk's body text. parse_markdown should drop all of it.
FURNITURE_SAMPLE = """<!-- page: 1 -->
# প্রথম অংশ
## প্রথম অধ্যায়
### ১। সংক্ষিপ্ত শিরোনাম।—
<!-- gazette_page: ৯৫৬০ -->
বাংলাদেশ গেজেট, অতিরিক্ত, সেপ্টেম্বর ২৮, ২০২৩
(৯৫৫১)
মূল্য : টাকা ২২৪.০০
(১) এই বিধিমালা পাবলিক প্রকিউরমেন্ট বিধিমালা, ২০২৫ নামে অভিহিত হইবে।
১৬০৪ বাংলাদেশ গেজেট, অতিরিক্ত, সেপ্টেম্বর ২৮, ২০২৫
(২) ইহা অবিলম্বে কার্যকর হইবে।
<!-- page: 2 -->
"""


def test_gazette_furniture_is_stripped():
    sections = parse_markdown(FURNITURE_SAMPLE)
    rule1 = next(s for s in sections if s.title.startswith("১।"))
    text = rule1.text
    assert "gazette_page" not in text
    assert "বাংলাদেশ গেজেট" not in text
    assert "মূল্য" not in text
    assert "৯৫৫১" not in text
    # The real content on either side of the furniture must survive intact.
    assert "(১) এই বিধিমালা" in text
    assert "(২) ইহা অবিলম্বে কার্যকর হইবে।" in text


def test_subrule_reference_in_parens_is_not_stripped():
    # "(১০)" style sub-rule references (1-3 digits) must NOT be treated as
    # furniture, even though they look superficially like "(৯৫৫১)".
    sample = """<!-- page: 1 -->
# প্রথম অংশ
## প্রথম অধ্যায়
### ২। সংজ্ঞার্থ।—
(১০) 'কল অফ' অর্থ এই বিধিমালার অধীন পুনঃপুন ক্রয়।
"""
    sections = parse_markdown(sample)
    rule = next(s for s in sections if s.title.startswith("২।"))
    assert "(১০)" in rule.text
