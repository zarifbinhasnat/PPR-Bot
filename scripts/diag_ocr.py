"""Diagnostic: inspect the FULL Gemini response (finish_reason, usage,
prompt_feedback) for a page that returned empty text, to learn why."""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from google import genai
from google.genai import types

from ppr_bot.config import settings
from ppr_bot.generation.prompts import OCR_TRANSCRIPTION_PROMPT

client = genai.Client(api_key=settings.GEMINI_API_KEY)


def probe(page: int, model: str):
    img = settings.pages_images_dir / f"page_{page:03d}.png"
    part = types.Part.from_bytes(data=img.read_bytes(), mime_type="image/png")
    try:
        r = client.models.generate_content(model=model, contents=[part, OCR_TRANSCRIPTION_PROMPT])
        text_len = len(r.text or "")
        fr = None
        if r.candidates:
            fr = r.candidates[0].finish_reason
        print(f"page {page} [{model}] text_len={text_len} finish_reason={fr} usage={r.usage_metadata}")
        print(f"    prompt_feedback={r.prompt_feedback}")
    except Exception as e:
        print(f"page {page} [{model}] ERROR {type(e).__name__}: {str(e)[:300]}")


# page 61 returned empty on 3 models; page 15 was saved as 0 bytes.
for pg in (61, 15):
    probe(pg, "gemini-flash-latest")
