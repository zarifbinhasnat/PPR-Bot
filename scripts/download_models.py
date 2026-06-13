"""Download the local models (bge-m3 embedder + bge-reranker-v2-m3) WITHOUT
relying on huggingface_hub's downloader.

Why this exists: on some networks huggingface_hub's chunked transfer protocol
(Xet / hf_transfer / its internal client) hangs at 0 bytes even though plain
authenticated HTTPS Range requests to the same CDN work fine (~1 MB/s in
testing). So this script fetches each file itself with `requests`, streaming to
disk with **resume** (HTTP Range) and **retries** — robust to a flaky link —
and saves into data/models/<name>/ so SentenceTransformer/CrossEncoder can load
them by local path (see settings.embedding_model_ref / reranker_model_ref).

We deliberately skip the ONNX weights (another ~2.3 GB we don't use on CPU) and
the docs/images, downloading only what sentence-transformers needs.

Auth: set HF_TOKEN in the environment (these models are public, but an
authenticated request avoids HF's anonymous large-file throttling).

Usage:
    HF_TOKEN=hf_xxx venv/Scripts/python.exe scripts/download_models.py
Safe to re-run: completed files are skipped, partial files resume.
"""

import os
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ppr_bot.config import settings

TOKEN = os.environ.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
CHUNK = 1 << 20  # 1 MiB
READ_TIMEOUT = 30  # seconds; a stalled read aborts and we resume

# (repo_id, local_subdir). Files are filtered by _wanted() below.
MODELS = [
    ("BAAI/bge-m3", "bge-m3"),
    ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
]

_SKIP_PREFIX = ("onnx/", "imgs/", "assets/")
_SKIP_SUFFIX = (".jpg", ".jpeg", ".png", ".webp", ".gitattributes", ".ds_store")


def _wanted(path: str) -> bool:
    """True for the structural files sentence-transformers needs (skip the
    ONNX duplicate weights and the doc images)."""
    low = path.lower()
    if low.startswith(_SKIP_PREFIX):
        return False
    if low.endswith(_SKIP_SUFFIX):
        return False
    if low == "readme.md":
        return False
    return True


def _list_files(repo: str) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi

    api = HfApi(token=TOKEN or None)
    info = api.model_info(repo, files_metadata=True)
    return [(s.rfilename, s.size or 0) for s in info.siblings if _wanted(s.rfilename)]


def _download_file(repo: str, rfilename: str, expected: int, dest: Path) -> None:
    """Stream one file to `dest` with resume + retry until its size matches."""
    url = f"https://huggingface.co/{repo}/resolve/main/{rfilename}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"      [skip] {rfilename} (already complete)")
        return

    attempt = 0
    while True:
        attempt += 1
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have == expected:
            return
        headers = dict(HEADERS)
        mode = "wb"
        if have > 0:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=(15, READ_TIMEOUT)
            ) as r:
                r.raise_for_status()
                with open(dest, mode) as f:
                    for block in r.iter_content(CHUNK):
                        if block:
                            f.write(block)
            size = dest.stat().st_size
            if not expected or size == expected:
                mb = size / 1e6
                print(f"      [done] {rfilename} ({mb:.1f} MB)")
                return
            print(f"      [partial] {rfilename}: {size}/{expected} bytes, resuming")
        except Exception as exc:
            wait = min(30, 3 * attempt)
            got = dest.stat().st_size if dest.exists() else 0
            print(
                f"      [retry {attempt}] {rfilename}: {type(exc).__name__} "
                f"at {got}/{expected or '?'} bytes — retrying in {wait}s"
            )
            time.sleep(wait)


def _download_repo(repo: str, subdir: str) -> Path:
    target = settings.models_dir / subdir
    print(f"\n=== {repo} -> {target} ===")
    files = _list_files(repo)
    total = sum(sz for _, sz in files)
    print(f"  {len(files)} files, ~{total/1e6:.0f} MB (ONNX/docs skipped)")
    # Big files last so the small config/tokenizer files land first.
    for rfilename, size in sorted(files, key=lambda x: x[1]):
        _download_file(repo, rfilename, size, target / rfilename)
    return target


def main() -> None:
    if not TOKEN:
        print("WARNING: HF_TOKEN not set — anonymous downloads may be throttled.")
    t0 = time.time()
    for repo, subdir in MODELS:
        _download_repo(repo, subdir)

    # Verify by actually loading each model from its local directory.
    print("\n=== Verifying local load ===")
    from sentence_transformers import CrossEncoder, SentenceTransformer

    emb = SentenceTransformer(str(settings.models_dir / "bge-m3"), device="cpu")
    dim = emb.encode(["পরীক্ষা / test"], normalize_embeddings=True).shape[1]
    print(f"  embedder OK — dim={dim}")

    rr = CrossEncoder(str(settings.models_dir / "bge-reranker-v2-m3"), device="cpu")
    score = float(rr.predict([("test query", "test document")])[0])
    print(f"  reranker OK — score={score:.3f}")

    print(f"\nDone in {time.time() - t0:.0f}s. Models cached under {settings.models_dir}")


if __name__ == "__main__":
    main()
