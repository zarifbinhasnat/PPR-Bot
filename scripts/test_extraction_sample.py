"""Milestone 2 verification: OCR a handful of representative pages.

Run this BEFORE the full 271-page extraction (run_extraction.py) to sanity
check transcription quality across the document's different page types:

  - page 1:   gazette cover page (legacy SutonnyMJ Bangla font)
  - page 6:   normal Bangla legal text (legacy font)
  - page 50:  normal Bangla legal text, mid-document
  - page 200: image-heavy table page (Annual Procurement Plan form)
  - page 225: scanned flowchart/diagram (Open Tendering Method)
  - page 251: English Standard Tender Document text

Usage (from project root, with venv activated):
    python scripts/test_extraction_sample.py
"""

import sys
from pathlib import Path

# Windows consoles often default to a non-UTF-8 codepage (e.g. cp932/cp1252)
# that can't encode Bangla characters, which would crash on print(). Force
# UTF-8 on stdout so we can print the transcriptions safely.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow running this script directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google import genai

from ppr_bot.config import settings
from ppr_bot.ingestion.ocr_transcriber import transcribe_page
from ppr_bot.ingestion.pdf_renderer import render_pdf_to_images

SAMPLE_PAGES = [1, 6, 50, 200, 225, 251]


def main() -> None:
    if not settings.GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is empty. Add it to .env before running this script."
        )

    print(f"Rendering sample pages {SAMPLE_PAGES} ...")
    image_paths = render_pdf_to_images(
        settings.pdf_path, settings.pages_images_dir, dpi=150, pages=SAMPLE_PAGES
    )

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    out_dir = settings.pages_markdown_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path, page_number in zip(image_paths, SAMPLE_PAGES):
        print(f"\n{'=' * 60}\nPage {page_number} ({image_path.name})\n{'=' * 60}")
        markdown = transcribe_page(image_path, client, settings.GEMINI_OCR_MODEL)
        print(markdown)

        out_path = out_dir / f"page_{page_number:03d}.md"
        out_path.write_text(markdown, encoding="utf-8")
        print(f"\n[saved to {out_path}]")


if __name__ == "__main__":
    main()
