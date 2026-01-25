import pytest
from unittest.mock import MagicMock
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory import (
    AgentFactory,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine import (
    ArxivAugmentedEngine,
)


def test_agent_factory_create_agent():
    """Test AgentFactory.create_agent returns correct agent types."""
    factory = AgentFactory()

    # Test research agent
    # Note: We might need to mock ArxivAugmentedEngine if it tries to init LLM
    # But for now let's see if we can just check the type if it's already initialized or mock it.

    # Mocking the internal _research_agent to avoid actual initialization
    mock_agent = MagicMock(spec=ArxivAugmentedEngine)
    factory._research_agent = mock_agent

    # Test research type
    agent = factory.create_agent("research")
    assert agent == mock_agent

    # Test basic type (currently maps to same agent in our implementation)
    agent_basic = factory.create_agent("basic")
    assert agent_basic == mock_agent

    # Test unknown type
    with pytest.raises(ValueError, match="Unknown agent type"):
        factory.create_agent("unknown")


def test_agent_factory_get_chat_agent():
    """Test get_chat_agent returns the research agent."""
    factory = AgentFactory()
    mock_agent = MagicMock(spec=ArxivAugmentedEngine)
    factory._research_agent = mock_agent

    assert factory.get_chat_agent() == mock_agent
