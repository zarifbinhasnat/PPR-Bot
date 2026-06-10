# 07 · FastAPI & Server-Sent Events (SSE)

## FastAPI in three ideas

1. **Typed request models.** `api/schemas.py` declares `ChatRequest` with
   Pydantic. FastAPI validates incoming JSON against it *before* your handler
   runs, and auto-generates interactive docs at `/docs`.

2. **Lifespan startup.** Loading the embedder + reranker + indexes takes
   seconds and gigabytes. Doing that per request would be catastrophic. The
   `lifespan` context manager in `api/main.py` loads them **once** at boot and
   stores them on `app.state`; every request reuses the same objects. (If the
   indexes aren't built yet, it boots in NOT-READY mode instead of crashing.)

3. **Routers.** Endpoints live in `routes_chat.py` and `routes_health.py` as
   `APIRouter`s, included into the app. Keeps things modular.

## Why SSE for streaming

When the LLM generates an answer token-by-token, we want tokens to appear in
the browser as they're produced — not after the whole answer is ready. We need
**server → client streaming**.

- **WebSockets** are bidirectional and heavier than we need.
- **Server-Sent Events (SSE)** are one-directional (server → client) over plain
  HTTP, which is exactly our case. The server keeps the response open and
  writes text frames:

  ```
  event: token
  data: {"type":"token","text":"Under "}

  event: token
  data: {"type":"token","text":"Rule 76 "}

  event: done
  data: {"type":"done"}
  ```

  (frames separated by a blank line)

## In this codebase

- **Server**: `routes_chat.py` returns `EventSourceResponse` (from
  `sse-starlette`) wrapping an async generator. The orchestrator is
  synchronous (LLM SDK + CPU models), so each step is pumped via
  `run_in_executor` to avoid blocking the event loop, and we check
  `request.is_disconnected()` to stop early if the user leaves.
- **Client**: `frontend/static/js/chat.js`. Note the browser's built-in
  `EventSource` only does **GET** with no body; our `/chat` is a **POST** with
  a JSON body, so we use `fetch()` + a `ReadableStream` reader and parse the
  SSE frames ourselves (`parseSSE`). Doing it by hand is a good way to *see*
  the wire format.
