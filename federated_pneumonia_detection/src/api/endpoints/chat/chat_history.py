"""Chat history endpoints."""

from __future__ import annotations

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException

from federated_pneumonia_detection.src.api.endpoints.schema import ChatHistoryResponse
from federated_pneumonia_detection.src.api.deps import get_query_engine
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems import (
    get_agent_factory,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """Retrieve conversation history for a session."""
    if get_query_engine() is None:
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    try:
        agent = get_agent_factory().get_chat_agent()
        history = agent.history(session_id)
        formatted_history = [
            {"user": user_msg, "assistant": ai_msg} for user_msg, ai_msg in history
        ]
        return ChatHistoryResponse(history=formatted_history, session_id=session_id)
    except Exception as exc:
        logger.error("Error retrieving history: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Error retrieving history: {str(exc)}"
        )


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str) -> Dict[str, str]:
    """Clear conversation history for a session."""
    if get_query_engine() is None:
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    try:
        agent = get_agent_factory().get_chat_agent()
        agent.clear_history(session_id)
        return {"message": f"History cleared for session {session_id}"}
    except Exception as exc:
        logger.error("Error clearing history: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Error clearing history: {str(exc)}"
        )
