"""Central configuration for PPR-Bot.

Every other module reads its settings from the single `settings` object
created here, instead of calling `os.getenv` directly. This keeps all
tunable values in one place (and in `.env`), which is the standard
"12-factor app" pattern: code stays the same across environments, only
the `.env` file changes.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/ppr_bot/config.py -> PPR-Bot/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Gemini API ---
    GEMINI_API_KEY: str = ""
    GEMINI_OCR_MODEL: str = "gemini-flash-latest"
    GEMINI_CHAT_MODEL: str = "gemini-flash-latest"
    GEMINI_AUX_MODEL: str = "gemini-flash-latest"

    # --- Local models ---
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

    # --- Retrieval tuning ---
    TOP_K_DENSE: int = 20
    TOP_K_SPARSE: int = 20
    TOP_K_RERANK: int = 5
    RRF_K: int = 60

    # --- Chunking ---
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64

    # --- Paths (derived, not from .env) ---
    @property
    def pdf_path(self) -> Path:
        return PROJECT_ROOT / "PPR 2025.pdf"

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def pages_images_dir(self) -> Path:
        return self.data_dir / "pages_images"

    @property
    def pages_markdown_dir(self) -> Path:
        return self.data_dir / "pages_markdown"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def full_document_path(self) -> Path:
        return self.processed_dir / "full_document.md"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def embeddings_path(self) -> Path:
        return self.processed_dir / "embeddings.npy"

    @property
    def bm25_index_path(self) -> Path:
        return self.processed_dir / "bm25_index.pkl"


# Single shared instance — `from ppr_bot.config import settings`
settings = Settings()
