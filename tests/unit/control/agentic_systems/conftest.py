import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_db_session():
    """MagicMock for SQLAlchemy session."""
    return MagicMock()


@pytest.fixture
def mock_llm():
    """MagicMock for LangChain LLM."""
    return MagicMock()


@pytest.fixture
def mock_vector_store():
    """MagicMock for PGVector."""
    return MagicMock()


@pytest.fixture
def sample_chat_input():
    """Sample ChatInput for testing."""
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.contracts import (
        ChatInput,
    )

    return ChatInput(query="test query", session_id="test_session")


@pytest.fixture
def mock_arxiv_engine():
    """Mock for ArxivAugmentedEngine."""
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.base_agent import (
        BaseAgent,
    )

    return MagicMock(spec=BaseAgent)


@pytest.fixture
def mock_base_tool():
    """Create a mock tool."""
    from unittest.mock import AsyncMock

    tool = AsyncMock()
    tool.name = "mock_tool"
    tool.description = "A mock tool"
    tool.ainvoke.return_value = "Tool result"
    return tool


@pytest.fixture
def mock_arxiv_tool():
    """Create a mock arxiv tool."""
    from unittest.mock import AsyncMock

    tool = AsyncMock()
    tool.name = "arxiv_search"
    tool.description = "Search Arxiv"
    tool.ainvoke.return_value = "Found 5 papers on federated learning"
    return tool
