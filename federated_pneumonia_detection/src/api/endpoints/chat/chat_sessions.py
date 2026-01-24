"""Chat session endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from federated_pneumonia_detection.src.api.endpoints.schema import (
    ChatSessionSchema,
    CreateSessionRequest,
)
from federated_pneumonia_detection.src.boundary.models.chat_session import ChatSession
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager import (
    SessionManager,
)

router = APIRouter()

# Singleton pattern ensures consistent history management across all endpoints
session_manager = SessionManager.get_instance()


@router.get("/sessions", response_model=List[ChatSessionSchema])
async def list_chat_sessions() -> List[ChatSessionSchema]:
    """List all available chat sessions."""
    sessions = session_manager.list_sessions()
    return [_to_schema(session) for session in sessions]


@router.post("/sessions", response_model=ChatSessionSchema)
async def create_new_chat_session(
    request: CreateSessionRequest = CreateSessionRequest(),
) -> ChatSessionSchema:
    """Create a new chat session with optional auto-generated title."""
    session = session_manager.create_session(
        title=request.title,
        initial_query=request.initial_query,
    )
    return _to_schema(session)


@router.delete("/sessions/{session_id}")
async def delete_existing_chat_session(session_id: str) -> dict:
    """Delete a chat session and clear its history."""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    session_manager.clear_history(session_id)
    return {"message": f"Session {session_id} deleted"}


def _to_schema(session: ChatSession) -> ChatSessionSchema:
    """Convert a ChatSession model to API schema."""
    return ChatSessionSchema(
        id=str(session.id),
        title=str(session.title),
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )
