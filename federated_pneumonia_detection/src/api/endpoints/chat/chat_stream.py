"""Chat streaming endpoint."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from federated_pneumonia_detection.src.api.endpoints.chat.chat_utils import (
    prepare_enhanced_query,
    sse_error,
    sse_pack,
)
from federated_pneumonia_detection.src.api.endpoints.schema import ChatMessage
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems import (
    AgentEventType,
    ChatInput,
    get_agent_factory,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager import (
    SessionManager,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton pattern ensures consistent history management across all endpoints
session_manager = SessionManager.get_instance()


@router.post("/query/stream")
async def query_chat_stream(message: ChatMessage, request: Request):
    """Stream chat response token by token using SSE."""
    logger.info(
        "[STREAM] Query: '%s...', Session: %s, Run: %s, Arxiv: %s",
        message.query[:100],
        message.session_id,
        message.run_id,
        message.arxiv_enabled,
    )

    async def generate():
        session_id = message.session_id or str(uuid.uuid4())
        yield sse_pack({"type": AgentEventType.SESSION.value, "session_id": session_id})

        session_manager.ensure_session(session_id, message.query)
        enhanced_query = await prepare_enhanced_query(message.query, message.run_id)

        chat_input = ChatInput(
            query=enhanced_query,
            session_id=session_id,
            arxiv_enabled=message.arxiv_enabled,
            run_id=message.run_id,
            original_query=message.query,
        )

        try:
            # Get factory with app.state to use pre-initialized engines
            agent_factory = get_agent_factory(app_state=request.app.state)
            agent = agent_factory.get_chat_agent()
        except Exception as exc:
            logger.error("[STREAM] Failed to initialize chat agent: %s", exc)
            yield sse_error(f"Failed to initialize engine: {exc}")
            return

        chunk_count = 0
        async for event in agent.stream(chat_input):
            chunk_count += 1
            yield sse_pack(event)

        if chunk_count == 0:
            logger.warning("[STREAM] No chunks generated - empty response")
            yield sse_error("No response generated from the query engine")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
