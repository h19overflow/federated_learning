"""Chat history endpoints."""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Request

from federated_pneumonia_detection.src.api.deps import get_query_engine
from federated_pneumonia_detection.src.api.endpoints.schema import ChatHistoryResponse
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems import (
    get_agent_factory,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    request: Request = None,
) -> ChatHistoryResponse:
    """
    Retrieve conversation history for a session with pagination.

    Args:
        session_id: Unique session identifier
        limit: Maximum number of conversation turns to return (default: 50)
        offset: Number of conversation turns to skip (default: 0)
        request: FastAPI request object for accessing app.state

    Returns:
        ChatHistoryResponse with paginated history
    """
    if get_query_engine() is None:
        raise HTTPException(
            status_code=500,
            detail="QueryEngine not initialized properly",
        )

    try:
        # Use pre-initialized factory from app.state (fallback to on-demand creation)
        agent_factory = (
            getattr(request.app.state, "agent_factory", None) if request else None
        )
        if agent_factory is None:
            logger.warning(
                "[HISTORY] No cached factory in app.state, creating on-demand"
            )
            agent_factory = get_agent_factory(
                app_state=request.app.state if request else None
            )
        agent = agent_factory.get_chat_agent()

        # Get full history
        full_history = agent.history(session_id)
        total_count = len(full_history)

        # Apply pagination: skip 'offset' items, take 'limit' items
        paginated_history = full_history[offset : offset + limit]

        formatted_history = [
            {"user": user_msg, "assistant": ai_msg}
            for user_msg, ai_msg in paginated_history
        ]

        return ChatHistoryResponse(
            history=formatted_history,
            session_id=session_id,
            total_count=total_count,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("Error retrieving history: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving history: {str(exc)}",
        )
