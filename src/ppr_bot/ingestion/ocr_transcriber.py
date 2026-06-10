"""Step 2 of the ingestion pipeline: OCR a rendered page image with Gemini.

Each page PNG (produced by `pdf_renderer.py`) is sent to a Gemini vision
model with `OCR_TRANSCRIPTION_PROMPT`, and we get back clean Bangla/English
Markdown. This single step replaces what would otherwise be two separate
pipelines: a legacy-font (SutonnyMJ) Unicode converter for the garbled text
pages, AND a Tesseract OCR pipeline for the scanned annexure pages — Gemini's
vision model "reads" the rendered page like a human, regardless of font or
scan status.

Robustness note: Gemini's free/shared model endpoints periodically return
HTTP 503 "high demand". Over a 271-page sequential job that's almost
guaranteed to hit some pages, so we (a) retry with backoff and (b) fall back
to alternate model names when the primary keeps failing. The alternates are
other capable vision models that tend to be overloaded at different times.
"""

import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from ppr_bot.generation.prompts import OCR_TRANSCRIPTION_PROMPT

# A single PDF page of this gazette transcribes to roughly 1-8 KB of Markdown.
# Anything dramatically larger is almost always a model "repetition loop"
# (the same character/word emitted thousands of times) — a known failure mode
# of smaller/cheaper models on dense non-Latin script. We reject such output
# so it never gets saved as if it were a real transcription.
_MAX_REASONABLE_CHARS = 25_000
# Matches any character repeated 40+ times in a row (e.g. "ôôôô…").
_REPEAT_RUN_RE = re.compile(r"(.)\1{39,}")


def _looks_degenerate(text: str) -> str | None:
    """Return a reason string if `text` looks like garbage, else None."""
    if len(text) > _MAX_REASONABLE_CHARS:
        return f"suspiciously long ({len(text)} chars) — likely a repetition loop"
    if _REPEAT_RUN_RE.search(text):
        return "contains a long single-character run — likely a repetition loop"
    return None

# Tried in order. If the configured primary model is overloaded OR has run out
# of its free daily quota, we fall through to these. Ordering matters and was
# determined empirically (scripts/diag_quota.py):
#   - The "-latest" aliases now point to brand-new premium models
#     (e.g. gemini-flash-latest -> gemini-3.5-flash) whose FREE-tier daily cap
#     is only ~20 requests/day — useless for a 271-page job.
#   - The "*-flash-lite" models have a far higher free daily cap (~1000/day),
#     so they are the workhorses here.
# All are vision-capable; the local bge embed/rerank models are unaffected.
FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]


def transcribe_page(
    image_path: Path,
    client: genai.Client,
    model: str,
    max_retries: int = 4,
    retry_delay_seconds: float = 6.0,
) -> str:
    """Send one page image to Gemini and return its Markdown transcription.

    Strategy: build a model list = [primary] + fallbacks (de-duplicated).
    For each model, retry up to `max_retries` times with linear backoff.
    Move to the next model only after the current one's retries are
    exhausted. Raise only if every model fails.
    """
    image_bytes = image_path.read_bytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    # Primary first, then fallbacks, skipping duplicates.
    models_to_try: list[str] = [model]
    for fb in FALLBACK_MODELS:
        if fb not in models_to_try:
            models_to_try.append(fb)

    last_error: Exception | None = None
    for model_name in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[image_part, OCR_TRANSCRIPTION_PROMPT],
                )
                text = (response.text or "").strip()
                # An empty 200-OK response is NOT success: Gemini occasionally
                # returns no text (transient hiccup / soft throttle / safety
                # filter). If we returned "" here, the orchestrator would save a
                # 0-byte page and mark it "done" — silently losing that page.
                # So treat empty as a failure and retry / fall through instead.
                if not text:
                    raise RuntimeError("empty transcription (no text in response)")
                degenerate = _looks_degenerate(text)
                if degenerate:
                    raise RuntimeError(f"degenerate transcription: {degenerate}")
                return text
            except Exception as exc:  # broad: covers SDK + transport errors
                last_error = exc
                if attempt < max_retries:
                    time.sleep(retry_delay_seconds * attempt)

    raise RuntimeError(
        f"Failed to transcribe {image_path.name} after trying "
        f"{models_to_try} ({max_retries} attempts each)"
    ) from last_error
