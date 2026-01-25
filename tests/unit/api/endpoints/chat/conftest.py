import pytest
from unittest.mock import MagicMock, patch
import sys


@pytest.fixture
def mock_orchestrator():
    """Mock StreamingOrchestrator (Agent)."""
    mock = MagicMock()

    # Mock the stream method which is an async generator
    async def mock_stream(*args, **kwargs):
        yield {"type": "session", "session_id": "test-session-id"}
        yield {"type": "token", "content": "Hello"}
        yield {"type": "token", "content": " world"}
        yield {"type": "done"}

    mock.stream = mock_stream
    mock.history.return_value = [("User message", "AI response")]
    return mock


@pytest.fixture
def mock_agent_factory(mock_orchestrator):
    """Mock AgentFactory."""
    mock = MagicMock()
    mock.get_chat_agent.return_value = mock_orchestrator
    return mock


@pytest.fixture
def mock_session_manager():
    """Mock SessionManager."""
    mock = MagicMock()
    mock.list_sessions.return_value = []

    # Create a mock session object
    mock_session = MagicMock()
    mock_session.id = "test-session-id"
    mock_session.title = "Test Session"
    mock_session.created_at.isoformat.return_value = "2023-01-01T00:00:00"
    mock_session.updated_at.isoformat.return_value = "2023-01-01T00:00:00"

    mock.create_session.return_value = mock_session
    mock.delete_session.return_value = True
    mock.get_session_history.return_value = [("User message", "AI response")]
    return mock


@pytest.fixture
def client(mock_agent_factory, mock_session_manager):
    """FastAPI TestClient with mocks."""
    from fastapi.testclient import TestClient

    # Import app here to avoid top-level side effects
    from federated_pneumonia_detection.src.api.main import app

    # Mock dependencies using dependency_overrides where possible
    from federated_pneumonia_detection.src.api.deps import (
        get_query_engine,
        get_mcp_manager,
    )

    mock_query_engine = MagicMock()
    app.dependency_overrides[get_query_engine] = lambda: mock_query_engine
    app.dependency_overrides[get_mcp_manager] = lambda: MagicMock()

    # For non-dependency-injected components, use patching
    # We patch the modules where these are used
    with (
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_sessions.session_manager",
            mock_session_manager,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_stream.session_manager",
            mock_session_manager,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_stream.get_agent_factory",
            return_value=mock_agent_factory,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_history.get_agent_factory",
            return_value=mock_agent_factory,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_history.get_query_engine",
            return_value=mock_query_engine,
        ),
    ):
        # Set app state
        app.state.agent_factory = mock_agent_factory
        app.state.query_engine = mock_query_engine

        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()
