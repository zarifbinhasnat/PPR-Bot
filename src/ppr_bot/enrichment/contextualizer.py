"""Step 5: contextual enrichment (Anthropic's "Contextual Retrieval").

Problem this solves: a chunk like "(৩) ইহার মূল্য ১০ লক্ষ টাকার অধিক হইবে না"
is meaningless in isolation — which rule? about what? Raw embedding/keyword
search struggles to connect it to a query like "direct procurement threshold
for goods".

Fix: for each chunk, make a cheap LLM call that writes a 1-2 sentence blurb
situating the chunk in the document (which Rule, what topic). We prepend that
blurb to form the `indexed_text` that goes into BOTH the vector index and the
BM25 index. The original `text` is preserved for display/answering.

This is the single most impactful "SOTA" retrieval upgrade in this project
for relatively little code — at the cost of one LLM call per chunk.
"""

import time

from google import genai

from ppr_bot.generation.prompts import CONTEXTUALIZE_CHUNK_PROMPT


def generate_context(
    chunk_text: str,
    breadcrumb: str,
    client: genai.Client,
    model: str,
    max_retries: int = 4,
    retry_delay_seconds: float = 6.0,
) -> str:
    """Return a short situating blurb for one chunk (empty string on failure).

    Failure here is non-fatal: if the context call fails, we index the chunk
    without a blurb rather than abort the whole job.
    """
    prompt = CONTEXTUALIZE_CHUNK_PROMPT.format(
        breadcrumb=breadcrumb, chunk_text=chunk_text
    )
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return (response.text or "").strip()
        except Exception:
            if attempt < max_retries:
                time.sleep(retry_delay_seconds * attempt)
    return ""  # give up gracefully; chunk still gets indexed un-contextualized


def build_indexed_text(context_blurb: str, chunk_text: str) -> str:
    """Combine the context blurb and chunk text into the text we index."""
    if context_blurb:
        return f"{context_blurb}\n\n{chunk_text}"
    return chunk_text
