"""Research agent adapter for chat endpoints."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, cast

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.base_agent import (
    BaseAgent,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine import (
    ArxivAugmentedEngine,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.contracts import (
    AgentEvent,
    AgentEventType,
    ChatInput,
)

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """Adapter exposing ArxivAugmentedEngine via the BaseAgent contract."""

    def __init__(self, engine: Optional[ArxivAugmentedEngine] = None) -> None:
        self._engine = engine or ArxivAugmentedEngine()

    async def stream(self, chat_input: ChatInput) -> AsyncGenerator[AgentEvent, None]:
        """Stream agent events for the given chat input."""
        try:
            async for event in self._engine.query_stream(
                chat_input.query,
                chat_input.session_id,
                arxiv_enabled=chat_input.arxiv_enabled,
                original_query=chat_input.original_query,
            ):
                yield self._normalize_event(event)
        except Exception as exc:
            logger.error("[ResearchAgent] Stream error: %s", exc, exc_info=True)
            yield {"type": AgentEventType.ERROR.value, "message": str(exc)}

    async def query(self, chat_input: ChatInput) -> Dict[str, Any]:
        """Run a non-streaming query for the provided chat input."""
        return await self._engine.query(
            chat_input.query,
            chat_input.session_id,
            arxiv_enabled=chat_input.arxiv_enabled,
            original_query=chat_input.original_query,
        )

    def history(self, session_id: str) -> List[Tuple[str, str]]:
        """Return conversation history for a session."""
        return self._engine.get_history(session_id)

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._engine.clear_history(session_id)

    def _normalize_event(self, event: Dict[str, Any]) -> AgentEvent:
        """Validate and normalize engine events."""
        if not isinstance(event, dict) or "type" not in event:
            return {"type": AgentEventType.ERROR.value, "message": "Invalid event"}
        return cast(AgentEvent, event)
