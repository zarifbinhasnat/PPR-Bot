"""Step 11: the FastAPI application.

Two big FastAPI concepts on display here:

1. **Lifespan startup loading.** The embedder, reranker, vector store, and
   BM25 index are expensive to load (seconds + GBs of RAM). Loading them on
   every request would make the API unusably slow. So we load them ONCE in
   the `lifespan` context manager when the server boots, stash them on
   `app.state`, and every request reuses them. (Loading models per-request is
   a classic beginner anti-pattern — we avoid it deliberately.)

2. **Routers.** Endpoints are grouped into `APIRouter`s in separate modules
   (routes_chat, routes_health) and included here, keeping the app modular.

Run it:  uvicorn ppr_bot.api.main:app --reload
Then open http://localhost:8000 for the chat UI, /docs for the API explorer.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ppr_bot.api import routes_chat, routes_health
from ppr_bot.chat.memory import InMemoryConversationStore
from ppr_bot.chat.orchestrator import ChatOrchestrator
from ppr_bot.config import settings
from ppr_bot.llm_client import get_client
from ppr_bot.retrieval.pipeline import RetrievalPipeline

# Force UTF-8 stdout so startup logs containing Bangla or punctuation like the
# em-dash don't crash on Windows' default cp932 console (the offline scripts do
# the same). Without this, a single "—" in a log line aborts server startup.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models/indexes once at startup; clean up at shutdown.

    If the offline pipeline hasn't been run yet (no embeddings/BM25 index on
    disk), we start anyway with `orchestrator = None` so /health works and
    /readiness honestly reports not-ready — instead of crashing on boot.
    """
    memory = InMemoryConversationStore()
    app.state.memory = memory
    app.state.orchestrator = None

    if settings.embeddings_path.exists() and settings.bm25_index_path.exists():
        print("Loading retrieval pipeline (embedder + reranker + indexes)...")
        pipeline = RetrievalPipeline()  # loads models + indexes (slow, once)
        client = get_client()
        app.state.orchestrator = ChatOrchestrator(pipeline, memory, client)
        print("Startup complete. Ready to serve.")
    else:
        print(
            "Index artifacts not found — starting in NOT-READY mode. "
            "Run the offline pipeline (extraction -> chunking -> enrichment "
            "-> indexing), then restart."
        )
    yield
    # (nothing to tear down for the in-memory store)


app = FastAPI(title="PPR-Bot", lifespan=lifespan)

# Permissive CORS for local development (frontend served from same origin,
# but this also allows calling the API from other local tools).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_chat.router)

# Serve the vanilla-JS chat UI at the root path.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
