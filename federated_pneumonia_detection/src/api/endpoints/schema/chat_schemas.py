"""Chat endpoint Pydantic schemas.

This module contains data models for chat API requests and responses,
separating schema definitions from business logic to follow SRP.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Request model for chat queries.

    Attributes:
        query: User question or query string.
        session_id: Optional session identifier for conversation continuity.
        run_id: Optional training run ID for context-aware responses.
        training_mode: Optional mode identifier (centralized/federated).
        arxiv_enabled: Whether to enable arxiv augmentation.
    """

    query: str
    session_id: Optional[str] = None
    run_id: Optional[int] = None
    training_mode: Optional[str] = None
    arxiv_enabled: bool = False


class ChatResponse(BaseModel):
    """Response model for chat queries.

    Attributes:
        answer: The generated response text.
        sources: List of source document references.
        session_id: Session identifier for this conversation.
    """

    answer: str
    sources: List[str] = []
    session_id: str


class ChatHistoryResponse(BaseModel):
    """Response model for chat history retrieval.

    Attributes:
        history: List of conversation turns (user message, assistant message).
        session_id: Session identifier for this conversation.
    """

    history: List[Dict[str, str]]
    session_id: str


class ChatSessionSchema(BaseModel):
    """Schema for chat session representation."""

    id: str
    title: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class CreateSessionRequest(BaseModel):
    """Request model for creating a new chat session."""

    title: Optional[str] = None
    initial_query: Optional[str] = None
