"""Step 1 of the ingestion pipeline: render PDF pages to images.

Why render to images instead of extracting the embedded text directly?
The source PDF (`PPR 2025.pdf`) mixes a legacy, non-Unicode Bangla font
(SutonnyMJ) with proper Unicode fonts, and has dozens of scanned pages with
no text layer at all. PyMuPDF's `get_text()` would return mojibake for the
SutonnyMJ pages and nothing for the scanned pages.

Instead, we render every page to a PNG and hand it to a vision-capable LLM
(see `ocr_transcriber.py`), which "reads" the page the way a human would —
this works uniformly regardless of the underlying font or whether the page
is a scan.
"""

from pathlib import Path

import fitz  # PyMuPDF


def render_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 150,
    pages: list[int] | None = None,
) -> list[Path]:
    """Render selected 1-indexed `pages` of `pdf_path` to PNGs in `output_dir`.

    If `pages` is None, every page in the document is rendered.
    Returns the written file paths in page order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        # PyMuPDF's default render is 72 DPI; scale up for OCR legibility.
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        page_numbers = pages if pages is not None else range(1, doc.page_count + 1)

        written: list[Path] = []
        for page_number in page_numbers:
            page = doc[page_number - 1]  # fitz pages are 0-indexed
            pix = page.get_pixmap(matrix=matrix)
            out_path = output_dir / f"page_{page_number:03d}.png"
            pix.save(out_path)
            written.append(out_path)

        return written
    finally:
        doc.close()
