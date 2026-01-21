"""Research mode streaming for the agent."""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Dict, List

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


async def stream_research(
    agent,
    messages: List,
    state: StreamState,
) -> AsyncGenerator[Dict[str, str], None]:
    """Stream research mode responses with tool reporting."""
    chunk_count = 0
    tools_reported = set()

    try:
        async for token, metadata in agent.astream(
            {"messages": messages},
            stream_mode="messages",
        ):
            chunk_count += 1
            node_name = metadata.get("langgraph_node", "")

            has_text = hasattr(token, "text") and token.text
            has_content = hasattr(token, "content") and token.content
            has_tool_chunks = (
                hasattr(token, "tool_call_chunks") and token.tool_call_chunks
            )

            if has_tool_chunks:
                for tool_chunk in token.tool_call_chunks:
                    tool_name = tool_chunk.get("name")
                    if tool_name and tool_name not in tools_reported:
                        tools_reported.add(tool_name)
                        friendly_name = tool_name.replace("_", " ").title()
                        yield create_sse_event(
                            SSEEventType.STATUS,
                            content=f"Using {friendly_name}...",
                        )
                        yield create_sse_event(
                            SSEEventType.TOOL_CALL,
                            tool=tool_name,
                            args={},
                        )
                        state.tool_calls.append(
                            {
                                "name": tool_name,
                                "args": tool_chunk.get("args", {}),
                            }
                        )

            if node_name == "tools":
                continue

            text_content = None
            if has_text:
                text_content = token.text
            elif has_content:
                text_content = normalize_content(token.content)

            if text_content:
                state.full_response += text_content
                yield create_sse_event(SSEEventType.TOKEN, content=text_content)

    except Exception as exc:
        logger.error("[ArxivEngine] Agent streaming failed: %s", exc, exc_info=True)
        yield create_sse_event(
            SSEEventType.ERROR, message=f"Agent execution failed: {str(exc)}"
        )
        return

    if not state.full_response:
        logger.warning(
            "[ArxivEngine] No response in agent stream. Chunks: %s",
            chunk_count,
        )
        yield create_sse_event(
            SSEEventType.ERROR,
            message="Agent completed but produced no response. Please try rephrasing.",
        )
