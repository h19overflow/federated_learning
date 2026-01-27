from unittest.mock import MagicMock, patch

import pytest

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager import (  # noqa: E501
    SessionManager,
)


@pytest.fixture
def mock_history_manager():
    with patch(
        "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.ChatHistoryManager"
    ) as mock:
        yield mock


@pytest.fixture
def mock_crud():
    with (
        patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session"
        ) as mock_create,
        patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_all_chat_sessions"
        ) as mock_get_all,
        patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_chat_session"
        ) as mock_get,
        patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.delete_chat_session"
        ) as mock_delete,
    ):
        yield {
            "create": mock_create,
            "get_all": mock_get_all,
            "get": mock_get,
            "delete": mock_delete,
        }


def test_session_manager_singleton():
    """Test that SessionManager is a singleton."""
    sm1 = SessionManager.get_instance()
    sm2 = SessionManager.get_instance()
    assert sm1 is sm2


def test_create_session(mock_crud):
    """Test session creation."""
    # Reset singleton for clean test if necessary, but here we just test the method
    sm = SessionManager.get_instance()

    mock_crud["create"].return_value = MagicMock(id="session-123")

    session = sm.create_session(title="Test Session")

    mock_crud["create"].assert_called_once_with(title="Test Session")
    assert session.id == "session-123"


def test_get_session_history():
    """Test retrieving session history."""
    sm = SessionManager.get_instance()

    # Mock the internal history manager
    mock_history = [("User query", "AI response")]
    sm._history_manager.get_history = MagicMock(return_value=mock_history)

    history = sm.get_session_history("session-123")

    sm._history_manager.get_history.assert_called_once_with("session-123")
    assert history == mock_history


def test_ensure_session(mock_crud):
    """Test ensure_session logic."""
    sm = SessionManager.get_instance()

    # Case 1: Session exists
    mock_crud["get"].return_value = MagicMock()
    sm.ensure_session("existing-id", "some query")
    mock_crud["create"].assert_not_called()

    # Case 2: Session does not exist
    mock_crud["get"].return_value = None
    sm.ensure_session("new-id", "some query")
    mock_crud["create"].assert_called()
