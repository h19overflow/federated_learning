import pytest
from unittest.mock import MagicMock, patch
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router import (
    classify_query,
    QueryClassification,
)


@pytest.fixture
def mock_llm():
    """Mock the router LLM instance."""
    with patch(
        "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm"
    ) as mock:
        llm_instance = MagicMock()
        mock.return_value = llm_instance
        yield llm_instance


def test_classify_query_research(mock_llm):
    """Scenario 1: LLM returns 'research' -> 'research'."""
    mock_llm.invoke.return_value = QueryClassification(mode="research")

    result = classify_query(
        "What is the impact of federated learning on pneumonia detection?"
    )

    assert result == "research"
    mock_llm.invoke.assert_called_once()


def test_classify_query_basic(mock_llm):
    """Scenario 2: LLM returns 'basic' -> 'basic'."""
    mock_llm.invoke.return_value = QueryClassification(mode="basic")

    result = classify_query("Hello, how are you today?")

    assert result == "basic"
    mock_llm.invoke.assert_called_once()


def test_classify_query_fallback(mock_llm):
    """Scenario 3: Low confidence/Fallback logic (Exception)."""
    mock_llm.invoke.side_effect = Exception("Connection error")

    # Should default to "research" on failure
    result = classify_query("Tell me about X-ray analysis")

    assert result == "research"
