"""Factory for multi-agent system components."""

from __future__ import annotations

from typing import Optional

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.base_agent import (
    BaseAgent,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.providers.research_agent import (
    ResearchAgent,
)


class AgentFactory:
    """Factory for initializing and reusing agents."""

    def __init__(self) -> None:
        self._research_agent: Optional[BaseAgent] = None

    def get_chat_agent(self) -> BaseAgent:
        """Return the primary chat agent for streaming queries."""
        if self._research_agent is None:
            self._research_agent = ResearchAgent()
        return self._research_agent


_agent_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Return the singleton AgentFactory instance."""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory()
    return _agent_factory
