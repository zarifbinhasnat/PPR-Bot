# 02 · Chunking strategies

A "chunk" is a unit of text you embed and retrieve. Chunking choices quietly
determine retrieval quality more than almost anything else.

## Why chunk at all?

- Embeddings represent a *fixed-size* meaning vector; a whole 270-page doc
  can't be one vector.
- Retrieval should return a *focused* passage, not a whole chapter, so the
  generator isn't drowned in irrelevant text.

## Naïve approach (what we avoid)

**Fixed-size windows**: split every N characters/tokens with some overlap.
Easy, but it slices mid-sentence and mid-rule, destroying structure. A chunk
might start "(৩) the limit shall be…" with no idea which rule "(৩)" belongs
to.

## Our approach: structure-aware chunking

PPR 2025 has a real hierarchy: Part → Chapter → Rule → Sub-rule, plus
Schedules. We exploit it (`chunking/markdown_parser.py` + `chunker.py`):

1. **Parse the OCR'd Markdown into a heading tree.** The OCR step was prompted
   to emit `#`/`##`/`###`/`####` for Part/Chapter/Rule/Schedule, so we can
   reconstruct the hierarchy.
2. **Use each leaf section (usually a Rule) as the natural chunk.**
3. **Split only when a section is too long**, on paragraph/sentence
   boundaries, with overlap so a fact spanning the cut is retrievable from
   either side.
4. **Attach metadata to every chunk**: a `breadcrumb` (e.g.
   `প্রথম অধ্যায় > ২। সংজ্ঞার্থ`), the `source_pages`, and the heading level.
   The breadcrumb is even prepended to the chunk text so it's self-describing.

## Why metadata matters

When the bot answers, it can cite "Rule 2, page 1" because that lived in the
chunk's metadata. Retrieval could also *filter* by it later (e.g. only
Schedules).

See `tests/test_chunker.py` for the hierarchy/metadata behaviour pinned down.
