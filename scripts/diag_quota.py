"""Verify the fix: transcribe a previously-empty page via the real
transcribe_page() using the new configured model + fallback chain."""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from google import genai
from ppr_bot.config import settings
from ppr_bot.ingestion.ocr_transcriber import transcribe_page

print("Configured OCR model:", settings.GEMINI_OCR_MODEL)
client = genai.Client(api_key=settings.GEMINI_API_KEY)
img = settings.pages_images_dir / "page_015.png"
md = transcribe_page(img, client, settings.GEMINI_OCR_MODEL)
print("page 15 transcription length:", len(md))
print("---- first 300 chars ----")
print(md[:300])
