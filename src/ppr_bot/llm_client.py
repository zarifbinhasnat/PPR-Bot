"""Single shared factory for the Gemini API client.

Why this file exists: every module that talks to Gemini (OCR extraction,
contextual enrichment, query transformation, answer generation) was
constructing its own `genai.Client(api_key=...)`. That's fine until you need
to change something for ALL of them at once — like adding a network timeout.

The bug this fixes: by default, the `google-genai` SDK's HTTP client has a
very long (or no) timeout. If a request gets stuck on a flaky/slow
connection, `generate_content()` can hang for many minutes with zero
feedback, making a "stuck" job indistinguishable from a "working" one. We saw
exactly this while running the 271-page OCR job.

The fix: pass `http_options=types.HttpOptions(timeout=...)` (milliseconds).
A request that exceeds this raises an exception instead of hanging — which
`transcribe_page()` / `generate_context()` already know how to retry or fall
back from.
"""

from google import genai
from google.genai import types

from ppr_bot.config import settings

# 60s is generous for a single vision-OCR or text-generation call, but finite.
DEFAULT_TIMEOUT_MS = 60_000


def get_client(timeout_ms: int = DEFAULT_TIMEOUT_MS) -> genai.Client:
    """Return a configured Gemini client with a bounded request timeout."""
    return genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
