"""Chat endpoints, including the Server-Sent Events (SSE) streaming chat.

Why SSE? When the LLM streams an answer token-by-token, we want those tokens
to appear in the browser as they arrive. SSE is the simplest standard for
server->client streaming over plain HTTP: the server holds the response open
and writes a sequence of `event:`/`data:` text frames. (WebSockets would be
overkill — we only need one direction.)

`sse-starlette`'s `EventSourceResponse` takes an async generator that yields
dicts like {"event": "...", "data": "..."} and handles the wire formatting,
keep-alives, and client-disconnect detection for us.

Each orchestrator event becomes one SSE message; the browser's reader parses
them by `event` type (query / citations / token / done).
"""

import asyncio
import json

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from ppr_bot.api.schemas import ChatRequest, HistoryResponse, Turn

router = APIRouter(tags=["chat"])


# Shown (as a normal streamed answer) when someone chats before the offline
# pipeline has produced an index, so the orchestrator isn't loaded yet.
_NOT_READY_MESSAGE = (
    "I'm not ready yet — the knowledge base is still being built "
    "(extraction → chunking → enrichment → indexing). Please try again once "
    "indexing has finished and the server has restarted."
)


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
    """Stream a grounded answer for `body.message` as SSE events."""
    orchestrator = request.app.state.orchestrator

    async def not_ready_stream():
        # Same event shape the client already knows, so the UI renders this as
        # an ordinary bot message instead of erroring on a dead stream.
        yield {"event": "token", "data": json.dumps({"type": "token", "text": _NOT_READY_MESSAGE}, ensure_ascii=False)}
        yield {"event": "done", "data": json.dumps({"type": "done"}, ensure_ascii=False)}

    if orchestrator is None:
        return EventSourceResponse(not_ready_stream())

    async def event_stream():
        # The orchestrator is synchronous (LLM SDK + CPU models). Run each
        # blocking step in a thread so we don't stall the event loop, while
        # still streaming results out as they're produced.
        loop = asyncio.get_event_loop()
        generator = orchestrator.handle_turn(
            body.session_id, body.message, rerank=body.rerank
        )

        def _next():
            # Pull one event from the sync generator (StopIteration -> None).
            return next(generator, None)

        while True:
            if await request.is_disconnected():
                break
            event = await loop.run_in_executor(None, _next)
            if event is None:
                break
            yield {
                "event": event["type"],
                "data": json.dumps(event, ensure_ascii=False),
            }

    return EventSourceResponse(event_stream())


@router.get("/chat/history/{session_id}", response_model=HistoryResponse)
def history(request: Request, session_id: str) -> HistoryResponse:
    """Return the stored turns for a session (e.g. to restore on page reload)."""
    memory = request.app.state.memory
    turns = [Turn(**t) for t in memory.get_history(session_id)]
    return HistoryResponse(session_id=session_id, turns=turns)
