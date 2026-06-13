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
    # These are HuggingFace repo ids by default. If the weights have been
    # pre-downloaded into data/models/<name>/ (see scripts/download_models.py),
    # the resolved *_model_ref properties below point at that local directory
    # instead, so loading works fully offline — useful when the HF Hub
    # downloader can't run (e.g. a network that stalls on its chunked
    # transfer protocol).
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
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def embedding_model_ref(self) -> str:
        """Local bge-m3 dir if its weights are present, else the HF repo id."""
        local = self.models_dir / "bge-m3"
        if (local / "pytorch_model.bin").exists():
            return str(local)
        return self.EMBEDDING_MODEL_NAME

    @property
    def reranker_model_ref(self) -> str:
        """Local reranker dir if its weights are present, else the HF repo id."""
        local = self.models_dir / "bge-reranker-v2-m3"
        if (local / "model.safetensors").exists():
            return str(local)
        return self.RERANKER_MODEL_NAME

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
