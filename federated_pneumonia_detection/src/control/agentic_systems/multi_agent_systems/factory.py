"""Factory for multi-agent system components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.base_agent import (  # noqa: E501
    BaseAgent,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine import (  # noqa: E501
    ArxivAugmentedEngine,
)

if TYPE_CHECKING:
    pass


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

            self._research_agent = (
                arxiv_engine if arxiv_engine is not None else ArxivAugmentedEngine()
            )
        return self._research_agent

    def create_agent(self, agent_type: str = "research") -> BaseAgent:
        """
        Create an agent of the specified type.

        Args:
            agent_type: Type of agent to create ('research' or 'basic')

        Returns:
            BaseAgent instance

        Raises:
            ValueError: If agent_type is unknown
        """
        if agent_type == "research":
            return self.get_chat_agent()
        elif agent_type == "basic":
            # Currently both use ArxivAugmentedEngine, but can be differentiated
            return self.get_chat_agent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


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
