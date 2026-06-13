"""Step 10b: the chat orchestrator — one place that runs a full chat turn.

This is the conductor that wires every component together for a single user
message. It's exactly the layer that frameworks like LangChain hide behind a
"chain" abstraction; we keep it explicit so the control flow is readable:

    1. read conversation history for this session
    2. condense the (possibly contextual) question into a standalone query
    3. retrieve + rerank the most relevant chunks for that query
    4. stream a grounded answer from the LLM
    5. persist both the user turn and the assistant turn to memory

It yields events as it goes so the API layer can forward them over SSE
without knowing anything about retrieval internals.
"""

from collections.abc import Iterator

from google import genai

from ppr_bot.chat.memory import ConversationStore
from ppr_bot.config import settings
from ppr_bot.generation.answer_generator import (
    generate_answer_stream,
    generate_suggestions,
)
from ppr_bot.retrieval.pipeline import RetrievalPipeline
from ppr_bot.retrieval.query_transform import condense_query

# Only the last few turns are needed to resolve a follow-up; keeping the
# history short controls token cost and keeps the condense prompt focused.
_HISTORY_WINDOW = 6


class ChatOrchestrator:
    def __init__(
        self,
        pipeline: RetrievalPipeline,
        memory: ConversationStore,
        client: genai.Client,
    ) -> None:
        self.pipeline = pipeline
        self.memory = memory
        self.client = client

    def handle_turn(
        self, session_id: str, message: str, rerank: bool = True
    ) -> Iterator[dict]:
        """Process one user message, yielding event dicts.

        Event shapes (consumed by the API/SSE layer):
            {"type": "query",      "query": <standalone query>}
            {"type": "citations",  "citations": [ {...}, ... ]}
            {"type": "token",      "text": <delta>}
            {"type": "suggestions","suggestions": [<str>, ...]}
            {"type": "done"}
        """
        history = self.memory.get_history(session_id)[-_HISTORY_WINDOW:]

        # 2. Condense follow-up into a standalone query.
        standalone = condense_query(
            history, message, self.client, settings.GEMINI_AUX_MODEL
        )
        yield {"type": "query", "query": standalone}

        # 3. Retrieve relevant chunks (rerank is toggleable from the UI).
        chunks = self.pipeline.retrieve(standalone, rerank=rerank)
        yield {
            "type": "citations",
            "citations": [
                {
                    "breadcrumb": c.get("metadata", {}).get("breadcrumb", ""),
                    "source_pages": c.get("metadata", {}).get("source_pages", []),
                    "rerank_score": c.get("rerank_score"),
                }
                for c in chunks
            ],
        }

        # 4. Stream the grounded answer, accumulating it for memory.
        answer_parts: list[str] = []
        for delta in generate_answer_stream(
            standalone, chunks, self.client, settings.GEMINI_CHAT_MODEL
        ):
            answer_parts.append(delta)
            yield {"type": "token", "text": delta}

        answer = "".join(answer_parts)

        # 5. Persist the turn (use the ORIGINAL user message in history).
        self.memory.append_turn(session_id, "user", message)
        self.memory.append_turn(session_id, "assistant", answer)

        # 6. Suggest a few natural follow-up questions for the UI's chips.
        #    Best-effort and cheap (AUX model); never blocks the answer above,
        #    and yields nothing useful if the daily quota is spent.
        suggestions = generate_suggestions(
            standalone, answer, self.client, settings.GEMINI_AUX_MODEL
        )
        if suggestions:
            yield {"type": "suggestions", "suggestions": suggestions}

        yield {"type": "done"}
