"""Factory for multi-agent system components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.base_agent import (
    BaseAgent,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.providers.research_agent import (
    ResearchAgent,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


class AgentFactory:
    """Factory for initializing and reusing agents."""

    def __init__(self, app_state=None) -> None:
        """
        Initialize the agent factory.

        Args:
            app_state: FastAPI app.state for accessing pre-initialized services
        """
        self._app_state = app_state
        self._research_agent: Optional[BaseAgent] = None

    def get_chat_agent(self) -> BaseAgent:
        """
        Return the primary chat agent for streaming queries.

        Uses pre-initialized engines from app.state if available for performance,
        otherwise falls back to lazy initialization.
        """
        if self._research_agent is None:
            # Try to get pre-initialized engines from app.state
            arxiv_engine = None
            if self._app_state is not None and hasattr(self._app_state, "arxiv_engine"):
                arxiv_engine = self._app_state.arxiv_engine

            self._research_agent = ResearchAgent(engine=arxiv_engine)
        return self._research_agent


_agent_factory: Optional[AgentFactory] = None


def get_agent_factory(app_state=None) -> AgentFactory:
    """
    Return the singleton AgentFactory instance.

    Args:
        app_state: FastAPI app.state for accessing pre-initialized services

    Returns:
        AgentFactory instance configured with app_state
    """
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory(app_state=app_state)
    return _agent_factory
