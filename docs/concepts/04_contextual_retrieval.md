# 04 · Contextual Retrieval

This is the technique introduced by Anthropic that gives the biggest
retrieval-quality boost in this project for the least code.

## The problem it solves

Chunks lose context when separated from their document. Consider:

> (৩) ইহার মূল্য ১০ লক্ষ টাকার অধিক হইবে না।
> ("(3) its value shall not exceed 10 lakh taka.")

In isolation this is nearly unsearchable. *Whose* value? Which rule? A user
asking "what is the threshold for direct procurement of goods?" will likely
never retrieve it — neither the embedding nor BM25 has anything to latch onto.

## The fix

Before indexing each chunk, ask an LLM to write a short blurb that **situates
the chunk in the whole document**, then **prepend it** to the chunk:

> এই অংশটি বিধি ৭৬ (সরাসরি ক্রয় পদ্ধতি) এর অধীনে পণ্য ক্রয়ের আর্থিক সীমা
> নির্ধারণ করে।
> (৩) ইহার মূল্য ১০ লক্ষ টাকার অধিক হইবে না।

Now both the **embedding** and the **BM25 tokens** carry "direct procurement
method", "goods", "financial threshold" — so the chunk becomes findable.

## In this codebase

- `enrichment/contextualizer.py` — `generate_context()` makes the LLM call;
  `build_indexed_text()` prepends the blurb.
- `enrichment/run_enrichment.py` — runs it over all chunks (resumable), adding
  `contextual_summary` and `indexed_text` to `chunks.jsonl`.
- `indexing/run_indexing.py` — indexes `indexed_text` (blurb + chunk), not the
  bare text. The original `text` is still what's shown/used at answer time.

## Cost note

It's one LLM call per chunk (~800–1200 calls). That's the second-biggest API
cost after OCR. The `--limit` flag lets you test on a sample first, and the
job resumes if interrupted.
