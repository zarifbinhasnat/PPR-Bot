"""Pydantic request/response models for the API.

FastAPI uses these for automatic request validation, and to generate the
interactive API docs at /docs. Declaring the shape of data at the boundary is
a core FastAPI idea — invalid requests are rejected before your code runs.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body of POST /chat."""

    message: str = Field(..., min_length=1, description="The user's question")
    session_id: str = Field(
        default="default",
        description="Conversation id; same id = continued conversation",
    )
    rerank: bool = Field(
        default=True,
        description=(
            "Whether to run the cross-encoder reranker. Off = faster but "
            "slightly less precise ordering (hybrid search only)."
        ),
    )


class Turn(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: list[Turn]
