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

import sys
import time

from google import genai

from ppr_bot.generation.prompts import CONTEXTUALIZE_CHUNK_PROMPT

# Tried in order, after the configured primary model. On a free-tier key,
# EACH Gemini model name has its OWN separate daily quota bucket (we
# discovered this the hard way: gemini-2.5-flash, gemini-flash-latest, AND
# gemini-2.5-flash-lite are each capped at 20 requests/day on this key — the
# earlier assumption that "*-lite" models had a ~1000/day cap was wrong).
# So a second model name effectively grants a second daily allowance for this
# step. Both names below are "lite" models, which is fine HERE: unlike OCR,
# contextualization is pure text generation (no Bangla glyphs to read), so
# the quality issues that rule lite models out for OCR
# (ingestion/ocr_transcriber.py) don't apply.
FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
]


def generate_context(
    chunk_text: str,
    breadcrumb: str,
    clients: list[genai.Client],
    model: str,
    max_retries: int = 4,
    retry_delay_seconds: float = 6.0,
) -> str:
    """Return a short situating blurb for one chunk (empty string on failure).

    Rotates through all configured API keys AND all fallback models before
    giving up — mirroring the OCR key-rotation pattern. Empty string on total
    exhaustion; the resume check in run_enrichment retries it next run.
    """
    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    prompt = CONTEXTUALIZE_CHUNK_PROMPT.format(
        breadcrumb=breadcrumb, chunk_text=chunk_text
    )

    last_error: Exception | None = None
    for key_idx, client in enumerate(clients):
        for model_name in models_to_try:
            for attempt in range(1, max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=model_name, contents=prompt
                    )
                    text = (response.text or "").strip()
                    if not text:
                        raise RuntimeError("empty contextualization response")
                    return text
                except Exception as exc:
                    last_error = exc
                    err_str = str(exc)
                    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                        # This model's quota on this key is gone — next model.
                        break
                    if attempt < max_retries:
                        time.sleep(retry_delay_seconds * attempt)
        if key_idx + 1 < len(clients):
            print(
                f"[enrich] key {key_idx + 1} exhausted — switching to key {key_idx + 2}.",
                file=sys.stderr,
            )

    print(
        f"[warn] contextualization failed for chunk "
        f"({breadcrumb!r}, tried {len(clients)} key(s) × {models_to_try}): {last_error}",
        file=sys.stderr,
    )
    return ""  # give up gracefully; chunk still gets indexed un-contextualized


def build_indexed_text(context_blurb: str, chunk_text: str) -> str:
    """Combine the context blurb and chunk text into the text we index."""
    if context_blurb:
        return f"{context_blurb}\n\n{chunk_text}"
    return chunk_text
