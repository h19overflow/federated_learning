"""Basic mode streaming for research agent."""

from __future__ import annotations

import logging
from typing import AsyncGenerator, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.content import (
    normalize_content,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.streaming import (
    SSEEventType,
    create_sse_event,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.stream_state import (
    StreamState,
)

logger = logging.getLogger(__name__)


async def stream_basic(
    llm,
    messages: List[SystemMessage | HumanMessage | AIMessage],
    state: StreamState,
) -> AsyncGenerator[dict, None]:
    """Stream basic mode responses from the LLM."""
    chunk_count = 0
    try:
        async for chunk in llm.astream(messages):
            chunk_count += 1
            if hasattr(chunk, "content") and chunk.content:
                content = normalize_content(chunk.content)
                if content:
                    state.full_response += content
                    yield create_sse_event(SSEEventType.TOKEN, content=content)
    except Exception as exc:
        logger.error(
            "[ArxivEngine] Basic mode streaming failed: %s", exc, exc_info=True
        )
        yield create_sse_event(
            SSEEventType.ERROR, message=f"Streaming failed: {str(exc)}"
        )
        return

    if not state.full_response:
        logger.error(
            "[ArxivEngine] Basic mode produced no response. Chunks: %s",
            chunk_count,
        )
        yield create_sse_event(
            SSEEventType.ERROR, message="No response generated. Please try again."
        )
