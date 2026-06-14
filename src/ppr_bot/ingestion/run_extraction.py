"""Step 3 (orchestration): extract the WHOLE PDF to Markdown, resumably.

This ties together `pdf_renderer.render_pdf_to_images` (PDF page -> PNG) and
`ocr_transcriber.transcribe_page` (PNG -> Markdown via Gemini) to process all
271 pages, then concatenates the per-page Markdown into one
`full_document.md`.

Key design idea: **resumability**. A 271-call job WILL be interrupted
eventually (rate limit, crash, you hit Ctrl+C). So we:
  - write each page's result to its own file `data/pages_markdown/page_NNN.md`
  - track per-page status in a `_manifest.json`
  - skip pages already done when re-run with --resume
This is a fundamental data-engineering pattern: make long batch jobs
idempotent and restartable instead of all-or-nothing.

Usage (from project root):
    # Process everything, resuming any prior partial run:
    python -m ppr_bot.ingestion.run_extraction --resume

    # Process just a range (good for a first small test):
    python -m ppr_bot.ingestion.run_extraction --start 1 --end 10

    # Only (re)build full_document.md from already-extracted pages:
    python -m ppr_bot.ingestion.run_extraction --concat-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import fitz

from ppr_bot.config import settings
from ppr_bot.ingestion.ocr_transcriber import transcribe_page
from ppr_bot.ingestion.pdf_renderer import render_pdf_to_images
from ppr_bot.llm_client import get_client

# Force UTF-8 stdout so progress lines with Bangla don't crash on Windows.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _manifest_path() -> Path:
    return settings.pages_markdown_dir / "_manifest.json"


def _load_manifest() -> dict:
    path = _manifest_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    _manifest_path().write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _page_markdown_path(page_number: int) -> Path:
    return settings.pages_markdown_dir / f"page_{page_number:03d}.md"


def _is_quota_error(exc: Exception) -> bool:
    s = str(exc)
    return "RESOURCE_EXHAUSTED" in s or "429" in s


def extract_pages(start: int, end: int, resume: bool) -> None:
    """Render + OCR pages [start, end] (1-indexed, inclusive).

    Supports KEY ROTATION: when the active key's daily quota is exhausted on
    all models, we switch to the next configured key (settings.gemini_api_keys)
    and retry the same page. Only when every key is spent do we stop — leaving
    the job resumable for the next quota reset.
    """
    settings.pages_markdown_dir.mkdir(parents=True, exist_ok=True)
    keys = settings.gemini_api_keys or [settings.GEMINI_API_KEY]
    key_idx = 0
    client = get_client(api_key=keys[key_idx])
    print(f"Using API key 1 of {len(keys)}")
    manifest = _load_manifest()

    for page_number in range(start, end + 1):
        key = str(page_number)
        out_path = _page_markdown_path(page_number)

        # Skip only if the page is marked done AND actually has content.
        # Some earlier runs saved 0-byte files (empty model responses) yet
        # marked them "done"; treat those as not-done so --resume redoes them.
        if (
            resume
            and manifest.get(key) == "done"
            and out_path.exists()
            and out_path.stat().st_size > 0
        ):
            print(f"[skip] page {page_number} already done")
            continue

        # Render once (cheap, no quota); re-used across any key switches.
        try:
            (image_path,) = render_pdf_to_images(
                settings.pdf_path, settings.pages_images_dir, dpi=150,
                pages=[page_number],
            )
        except Exception as exc:
            manifest[key] = f"error: render {exc}"
            print(f"[FAIL] page {page_number}: render {exc}")
            _save_manifest(manifest)
            continue

        t0 = time.time()
        while True:  # transcribe, rotating keys on quota exhaustion
            try:
                markdown = transcribe_page(
                    image_path, client, settings.GEMINI_OCR_MODEL
                )
                out_path.write_text(markdown, encoding="utf-8")
                manifest[key] = "done"
                print(f"[ok]   page {page_number} ({time.time() - t0:.1f}s)")
                _save_manifest(manifest)
                break
            except Exception as exc:
                # Quota exhausted on the current key? Try the next key and
                # retry THIS page, rather than losing it.
                if _is_quota_error(exc) and key_idx + 1 < len(keys):
                    key_idx += 1
                    print(
                        f"\nKey {key_idx} of {len(keys)} exhausted — switching "
                        f"to key {key_idx + 1}."
                    )
                    client = get_client(api_key=keys[key_idx])
                    continue
                manifest[key] = f"error: {exc}"
                print(f"[FAIL] page {page_number}: {exc}")
                _save_manifest(manifest)
                if _is_quota_error(exc):
                    print(
                        "\nAll API keys' daily free quota appear exhausted. "
                        "Stopping. Re-run with --resume after the quota resets "
                        "(midnight US Pacific time)."
                    )
                    return
                break  # non-quota error: give up on this page, move on


def concat_full_document() -> None:
    """Concatenate all per-page Markdown into one document.

    Page boundaries are preserved as HTML comments (`<!-- page: N -->`) so
    that chunks can later cite the exact source page they came from.
    """
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(settings.pdf_path)
    page_count = doc.page_count
    doc.close()

    parts: list[str] = []
    missing: list[int] = []
    for page_number in range(1, page_count + 1):
        path = _page_markdown_path(page_number)
        if not path.exists():
            missing.append(page_number)
            continue
        parts.append(f"<!-- page: {page_number} -->\n")
        parts.append(path.read_text(encoding="utf-8").strip())
        parts.append("\n\n")

    settings.full_document_path.write_text("".join(parts), encoding="utf-8")
    print(f"\nWrote {settings.full_document_path} ({len(parts) // 3} pages)")
    if missing:
        print(f"WARNING: {len(missing)} page(s) missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PPR 2025 PDF to Markdown.")
    parser.add_argument("--start", type=int, default=1, help="first page (1-indexed)")
    parser.add_argument("--end", type=int, default=None, help="last page (inclusive)")
    parser.add_argument(
        "--resume", action="store_true", help="skip pages already marked done"
    )
    parser.add_argument(
        "--concat-only",
        action="store_true",
        help="only rebuild full_document.md from existing per-page files",
    )
    args = parser.parse_args()

    if not args.concat_only:
        if not settings.GEMINI_API_KEY:
            raise SystemExit("GEMINI_API_KEY is empty. Set it in .env first.")
        end = args.end
        if end is None:
            doc = fitz.open(settings.pdf_path)
            end = doc.page_count
            doc.close()
        print(f"Extracting pages {args.start}..{end} (resume={args.resume})")
        extract_pages(args.start, end, args.resume)

    concat_full_document()


if __name__ == "__main__":
    main()
