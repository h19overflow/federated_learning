"""
Tests for session manager module.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from federated_pneumonia_detection.src.boundary.models.chat_session import ChatSession
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager import (
    SessionManager,
)


class TestSessionManager:
    """Test SessionManager class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset SessionManager singleton before each test."""
        SessionManager._instance = None

    @pytest.fixture
    def mock_history_manager(self):
        """Create mock ChatHistoryManager."""
        manager = MagicMock()
        manager.clear_history = Mock()
        return manager

    @pytest.fixture
    def session_manager(self, mock_history_manager):
        """Create SessionManager with mocked dependencies."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.ChatHistoryManager",
            return_value=mock_history_manager,
        ):
            return SessionManager()

    @pytest.fixture
    def mock_chat_session(self):
        """Create mock ChatSession."""
        session = MagicMock(spec=ChatSession)
        session.session_id = "test_session_id"
        session.title = "Test Session"
        session.created_at = "2024-01-01"
        session.updated_at = "2024-01-01"
        return session

    def test_session_manager_initialization(self, mock_history_manager):
        """Test that SessionManager initializes correctly."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.ChatHistoryManager",
            return_value=mock_history_manager,
        ):
            manager = SessionManager()

            assert manager._history_manager is not None

    def test_list_sessions(self, session_manager, mock_chat_session):
        """Test listing all chat sessions."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_all_chat_sessions",
            return_value=[mock_chat_session],
        ) as mock_get_all:
            sessions = session_manager.list_sessions()

            assert isinstance(sessions, list)
            assert len(sessions) == 1
            assert sessions[0].session_id == mock_chat_session.session_id
            mock_get_all.assert_called_once()

    def test_list_sessions_empty(self, session_manager):
        """Test listing sessions when none exist."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_all_chat_sessions",
            return_value=[],
        ) as mock_get_all:
            sessions = session_manager.list_sessions()

            assert sessions == []
            mock_get_all.assert_called_once()

    def test_create_session_with_title(self, session_manager, mock_chat_session):
        """Test creating session with explicit title."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
            return_value=mock_chat_session,
        ) as mock_create:
            session = session_manager.create_session(title="Custom Title")

            assert session.session_id == mock_chat_session.session_id
            mock_create.assert_called_once_with(title="Custom Title")

    def test_create_session_without_title(self, session_manager, mock_chat_session):
        """Test creating session without title (auto-generated)."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
                return_value=mock_chat_session,
            ) as mock_create,
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.generate_chat_title",
                return_value="Generated Title",
            ) as mock_generate_title,
        ):
            session = session_manager.create_session(
                initial_query="What is federated learning?",
            )

            assert session.session_id == mock_chat_session.session_id
            mock_generate_title.assert_called_once_with(
                "What is federated learning?",
            )
            mock_create.assert_called_once_with(title="Generated Title")

    def test_create_session_without_title_or_query(
        self,
        session_manager,
        mock_chat_session,
    ):
        """Test creating session without title or initial query."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
                return_value=mock_chat_session,
            ) as mock_create,
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.generate_chat_title",
            ) as mock_generate_title,
        ):
            session = session_manager.create_session()

            assert session.session_id == mock_chat_session.session_id
            mock_generate_title.assert_not_called()  # No query to generate from
            mock_create.assert_called_once_with(title=None)

    def test_ensure_session_existing(self, session_manager, mock_chat_session):
        """Test ensuring session that already exists."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_chat_session",
                return_value=mock_chat_session,
            ) as mock_get,
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
            ) as mock_create,
        ):
            session_manager.ensure_session("existing_session", "Test query")

            # Should check for existing session
            mock_get.assert_called_once_with("existing_session")
            # Should not create new session
            mock_create.assert_not_called()

    def test_ensure_session_new(self, session_manager, mock_chat_session):
        """Test ensuring session that doesn't exist."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_chat_session",
                return_value=None,
            ) as mock_get,
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
                return_value=mock_chat_session,
            ) as mock_create,
        ):
            session_manager.ensure_session("new_session", "Test query")

            mock_get.assert_called_once_with("new_session")
            mock_create.assert_called_once_with(
                title="Test query...",
                session_id="new_session",
            )

    def test_delete_session(self, session_manager):
        """Test deleting a session."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.delete_chat_session",
            return_value=True,
        ) as mock_delete:
            result = session_manager.delete_session("session_123")

            assert result is True
            mock_delete.assert_called_once_with("session_123")

    def test_delete_session_not_found(self, session_manager):
        """Test deleting non-existent session."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.delete_chat_session",
            return_value=False,
        ) as mock_delete:
            result = session_manager.delete_session("non_existent")

            assert result is False
            mock_delete.assert_called_once_with("non_existent")

    def test_clear_history(self, session_manager, mock_history_manager):
        """Test clearing history for a session."""
        session_manager.clear_history("session_123")

        mock_history_manager.clear_history.assert_called_once_with("session_123")


class TestSessionManagerErrorHandling:
    """Test error handling in SessionManager."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset SessionManager singleton before each test."""
        SessionManager._instance = None

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager for error testing."""
        return SessionManager()

    def test_ensure_session_db_error_logged(self, session_manager):
        """Test that DB errors in ensure_session are logged (non-fatal)."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_chat_session",
                side_effect=Exception("Database connection failed"),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
            ) as mock_create,
        ):
            # Should not raise exception
            session_manager.ensure_session("session_123", "query")

            # Should NOT attempt to create session as fallback (per current implementation)
            mock_create.assert_not_called()

    def test_ensure_session_create_failure_logged(self, session_manager):
        """Test that create session failures in ensure_session are logged."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_chat_session",
                return_value=None,
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
                side_effect=Exception("Failed to create session"),
            ),
        ):
            # Should not raise exception
            session_manager.ensure_session("session_123", "query")

    def test_list_sessions_db_error(self, session_manager):
        """Test list_sessions with DB error."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_all_chat_sessions",
            side_effect=Exception("DB error"),
        ):
            with pytest.raises(Exception, match="DB error"):
                session_manager.list_sessions()


class TestSessionManagerIntegration:
    """Integration tests for SessionManager."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset SessionManager singleton before each test."""
        SessionManager._instance = None

    @pytest.fixture
    def mock_sessions(self):
        """Create list of mock sessions."""
        sessions = []
        for i in range(3):
            session = MagicMock(spec=ChatSession)
            session.session_id = f"session_{i}"
            session.title = f"Session {i}"
            session.created_at = f"2024-01-{i + 1:02d}"
            session.updated_at = f"2024-01-{i + 2:02d}"
            sessions.append(session)
        return sessions

    def test_list_multiple_sessions(self, mock_sessions):
        """Test listing multiple sessions."""
        manager = SessionManager()

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_all_chat_sessions",
            return_value=mock_sessions,
        ):
            sessions = manager.list_sessions()

            assert len(sessions) == 3
            assert sessions[0].session_id == "session_0"
            assert sessions[2].session_id == "session_2"

    def test_session_lifecycle(self):
        """Test complete session lifecycle."""
        manager = SessionManager()
        mock_session = MagicMock(spec=ChatSession)
        mock_session.session_id = "lifecycle_session"

        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
                return_value=mock_session,
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_chat_session",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.delete_chat_session",
                return_value=True,
            ),
        ):
            # Create session
            session = manager.create_session(title="Lifecycle Test")
            assert session.session_id == "lifecycle_session"

            # Ensure session exists
            manager.ensure_session("lifecycle_session", "test query")

            # Delete session
            result = manager.delete_session("lifecycle_session")
            assert result is True

    def test_clear_history_after_conversation(self):
        """Test clearing history after a conversation."""
        manager = SessionManager()
        # Mock the history manager
        manager._history_manager = MagicMock()
        mock_session = MagicMock(spec=ChatSession)
        mock_session.session_id = "test_session"

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
            return_value=mock_session,
        ):
            # Create session
            manager.create_session(title="Test")

            # Simulate conversation
            manager._history_manager.add_to_history(
                "test_session",
                "User message",
                "AI response",
            )

            # Clear history
            manager.clear_history("test_session")

            manager._history_manager.clear_history.assert_called_once_with(
                "test_session",
            )


class TestSessionManagerWithRealDependencies:
    """Test SessionManager with partial real dependencies."""

    def test_session_manager_with_mock_crud_functions(self):
        """Test SessionManager with mocked CRUD functions."""
        manager = SessionManager()

        # Mock CRUD functions
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_all_chat_sessions",
                return_value=[],
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.create_chat_session",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.get_chat_session",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.session_manager.delete_chat_session",
            ),
        ):
            # Test all methods
            assert manager.list_sessions() == []
            manager.create_session(title="Test")
            manager.ensure_session("session_1", "query")
            manager.delete_session("session_1")
            manager.clear_history("session_1")

            # Verify no exceptions raised
            assert True
