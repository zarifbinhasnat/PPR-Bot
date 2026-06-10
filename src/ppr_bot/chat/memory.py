"""Step 10a: conversation memory.

A chat needs to remember earlier turns so follow-up questions can be
condensed (see query_transform.py) and the model has continuity. We define a
small abstract interface (`ConversationStore`) and one concrete
implementation that keeps everything in a Python dict in RAM.

In-memory is perfect for development and a single-process server, but it's
lost on restart and not shared across processes. The interface is here so you
can later drop in a `SQLiteConversationStore` (or Redis, etc.) WITHOUT
touching the orchestrator or API — they only depend on the interface. That
"program to an interface" separation is the lesson here.
"""

from abc import ABC, abstractmethod


class ConversationStore(ABC):
    """Stores chat turns per session id."""

    @abstractmethod
    def get_history(self, session_id: str) -> list[dict]:
        """Return prior turns for a session as [{'role','content'}, ...]."""

    @abstractmethod
    def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Append one turn (role is 'user' or 'assistant')."""


class InMemoryConversationStore(ConversationStore):
    """Dict-backed store: {session_id: [ {role, content}, ... ]}.

    To upgrade to durable storage, implement `ConversationStore` with SQLite:
    create a `turns(session_id, idx, role, content)` table, INSERT on
    append_turn, and SELECT ordered by idx on get_history. The rest of the
    app stays unchanged.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict]] = {}

    def get_history(self, session_id: str) -> list[dict]:
        return list(self._sessions.get(session_id, []))

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": role, "content": content}
        )
