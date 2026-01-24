"""Streaming orchestration for the research agent."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_helpers import (
    build_messages,
    build_tools,
    create_research_agent,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.stream_basic import (
    stream_basic,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.stream_research import (
    stream_research,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.stream_state import (
    StreamState,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router import (
    classify_query,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.streaming import (
    SSEEventType,
    create_sse_event,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.prompts.research_prompts import (
    BASIC_MODE_SYSTEM_PROMPT,
    RESEARCH_MODE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


async def stream_query(
    llm,
    history_manager,
    rag_tool,
    query: str,
    session_id: str,
    arxiv_enabled: bool = False,
    original_query: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream query response token by token."""
    query_mode = classify_query(query)
    logger.info("[ArxivEngine] Query classified as: %s", query_mode)

    tools = []
    if query_mode == "research":
        tools = build_tools(rag_tool, arxiv_enabled)
        logger.info("[ArxivEngine] Retrieved %s tools", len(tools))
        if not tools:
            yield create_sse_event(SSEEventType.ERROR, message="No tools available")
            return

    state = StreamState()
    if query_mode == "basic":
        try:
            # Get formatted conversation history
            history_context = history_manager.format_for_context(session_id)

            # Inject history into basic mode system prompt
            if history_context:
                enriched_prompt = f"{BASIC_MODE_SYSTEM_PROMPT}\n\nCONVERSATION HISTORY:\n{history_context}"
            else:
                enriched_prompt = BASIC_MODE_SYSTEM_PROMPT

            messages = build_messages(
                history_manager,
                query,
                session_id,
                include_system=True,
                system_prompt=enriched_prompt,
            )
        except Exception as e:
            logger.error(
                "[ArxivEngine] Failed to build messages (basic): %s", e, exc_info=True
            )
            # Fallback: create messages without history
            messages = [
                SystemMessage(content=BASIC_MODE_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
        async for event in stream_basic(llm, messages, state):
            yield event
    else:
        yield create_sse_event(SSEEventType.STATUS, content="Analyzing your query...")
        agent = create_research_agent(llm, tools, RESEARCH_MODE_SYSTEM_PROMPT)
        try:
            messages = build_messages(
                history_manager,
                query,
                session_id,
                include_system=False,
                system_prompt=RESEARCH_MODE_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error(
                "[ArxivEngine] Failed to build messages (research): %s",
                e,
                exc_info=True,
            )
            # Fallback: create messages without history
            messages = [HumanMessage(content=query)]
        async for event in stream_research(agent, messages, state):
            yield event

    if not state.full_response:
        return

    try:
        history_query = original_query if original_query is not None else query
        history_manager.add_to_history(session_id, history_query, state.full_response)
        logger.info("[ArxivEngine] Saved to history. Session: %s", session_id)
    except Exception as exc:
        logger.error("[ArxivEngine] Failed to save history: %s", exc, exc_info=True)

    yield create_sse_event(SSEEventType.DONE, session_id=session_id)
