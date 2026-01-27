"""
Tests for PostgreSQL chat history manager.
"""

from unittest.mock import MagicMock, Mock, patch
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history import (  # noqa: E501
    ChatHistoryManager,
)


class TestChatHistoryManager:
    """Test ChatHistoryManager class."""

    @pytest.fixture
    def mock_postgres_history(self):
        """Create mock PostgresChatMessageHistory."""
        history = MagicMock()

        # Mock messages
        history.messages = [
            HumanMessage(content="User question 1"),
            AIMessage(content="AI response 1"),
            HumanMessage(content="User question 2"),
            AIMessage(content="AI response 2"),
        ]

        # Mock methods
        history.add_messages = Mock()
        history.clear = Mock()
        history.create_tables = Mock()

        return history

    @pytest.fixture
    def mock_connection(self):
        """Create mock psycopg connection."""
        conn = MagicMock()
        conn.cursor = MagicMock()
        conn.commit = Mock()
        conn.rollback = Mock()
        conn.close = Mock()
        return conn

    @pytest.fixture
    def history_manager(self):
        """Create ChatHistoryManager for testing."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=MagicMock(),
            ),
        ):
            return ChatHistoryManager(table_name="test_messages", max_history=10)

    def test_manager_initialization(self):
        """Test that ChatHistoryManager initializes correctly."""
        manager = ChatHistoryManager(table_name="test_table", max_history=5)

        assert manager.table_name == "test_table"
        assert manager.max_history == 5

    def test_manager_initialization_defaults(self):
        """Test that ChatHistoryManager has default values."""
        manager = ChatHistoryManager()

        assert manager.table_name == "message_store"
        assert manager.max_history == 10

    def test_get_postgres_history_with_uuid(self, mock_connection):
        """Test _get_postgres_history with UUID session_id."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=mock_connection,
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=MagicMock(),
            ) as mock_history_class,
        ):
            manager = ChatHistoryManager()
            uuid_session_id = str(UUID("12345678-1234-5678-1234-567812345678"))

            manager._get_postgres_history(uuid_session_id)

            # Should use UUID directly
            mock_history_class.assert_called_once()
            call_args = mock_history_class.call_args[0]
            assert call_args[1] == uuid_session_id

    def test_get_postgres_history_with_string(self, mock_connection):
        """Test _get_postgres_history with string session_id (converted to UUID)."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=mock_connection,
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=MagicMock(),
            ) as mock_history_class,
        ):
            manager = ChatHistoryManager()
            string_session_id = "session_123"

            manager._get_postgres_history(string_session_id)

            # Should convert string to UUID
            mock_history_class.assert_called_once()
            call_args = mock_history_class.call_args[0]
            # Should be a UUID (generated via UUID5)
            assert isinstance(call_args[1], str)
            assert UUID(call_args[1])

    def test_add_to_history(self, mock_postgres_history):
        """Test adding messages to history."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_postgres_history,
            ),
        ):
            manager = ChatHistoryManager()
            manager.add_to_history("session_1", "User message", "AI response")

            # Verify messages were added
            mock_postgres_history.add_messages.assert_called_once()
            added_messages = mock_postgres_history.add_messages.call_args[0][0]

            assert len(added_messages) == 2
            assert isinstance(added_messages[0], HumanMessage)
            assert added_messages[0].content == "User message"
            assert isinstance(added_messages[1], AIMessage)
            assert added_messages[1].content == "AI response"

    def test_get_history(self, mock_postgres_history):
        """Test retrieving history."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_postgres_history,
            ),
        ):
            manager = ChatHistoryManager()
            history = manager.get_history("session_1")

            # Should return list of tuples
            assert isinstance(history, list)
            assert len(history) == 2
            assert history[0] == ("User question 1", "AI response 1")
            assert history[1] == ("User question 2", "AI response 2")

    def test_get_history_empty(self):
        """Test getting empty history."""
        mock_history = MagicMock()
        mock_history.messages = []

        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_history,
            ),
        ):
            manager = ChatHistoryManager()
            history = manager.get_history("session_1")

            assert history == []

    def test_get_history_with_odd_messages(self):
        """Test getting history with odd number of messages."""
        # Simulate incomplete conversation
        mock_history = MagicMock()
        mock_history.messages = [
            HumanMessage(content="User message"),
            AIMessage(content="AI response"),
            HumanMessage(content="Another user message"),  # No AI response yet
        ]

        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_history,
            ),
        ):
            manager = ChatHistoryManager()
            history = manager.get_history("session_1")

            # Should only return complete pairs
            assert len(history) == 1
            assert history[0] == ("User message", "AI response")

    def test_clear_history(self, mock_postgres_history):
        """Test clearing history."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_postgres_history,
            ),
        ):
            manager = ChatHistoryManager()
            manager.clear_history("session_1")

            mock_postgres_history.clear.assert_called_once()

    def test_format_for_context(self, mock_postgres_history):
        """Test formatting history as context string."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_postgres_history,
            ),
        ):
            manager = ChatHistoryManager()
            context = manager.format_for_context("session_1")

            # Should be formatted correctly
            expected = (
                "User: User question 1\n"
                "Assistant: AI response 1\n\n"
                "User: User question 2\n"
                "Assistant: AI response 2"
            )
            assert context == expected

    def test_format_for_context_empty(self):
        """Test formatting empty history."""
        mock_history = MagicMock()
        mock_history.messages = []

        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_history,
            ),
        ):
            manager = ChatHistoryManager()
            context = manager.format_for_context("session_1")

            assert context == ""

    def test_get_messages(self, mock_postgres_history):
        """Test getting raw messages."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_postgres_history,
            ),
        ):
            manager = ChatHistoryManager()
            messages = manager.get_messages("session_1")

            # Should return raw messages
            assert messages == mock_postgres_history.messages
            assert len(messages) == 4


class TestChatHistoryManagerErrorHandling:
    """Test error handling in ChatHistoryManager."""

    def test_add_to_history_handles_errors(self):
        """Test that add_to_history handles database errors."""
        mock_history = MagicMock()
        mock_history.add_messages = Mock(side_effect=Exception("DB Error"))

        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_history,
            ),
        ):
            manager = ChatHistoryManager()

            with pytest.raises(Exception, match="DB Error"):
                manager.add_to_history("session_1", "User", "AI")

    def test_get_history_handles_connection_error(self):
        """Test that get_history handles connection errors."""
        mock_pool = MagicMock()
        mock_pool.getconn.side_effect = Exception("Connection failed")

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
            return_value=mock_pool,
        ):
            manager = ChatHistoryManager()

            with pytest.raises(Exception, match="Connection failed"):
                manager.get_history("session_1")

    def test_clear_history_handles_errors(self):
        """Test that clear_history handles errors."""
        mock_history = MagicMock()
        mock_history.clear = Mock(side_effect=Exception("Clear failed"))

        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_history,
            ),
        ):
            manager = ChatHistoryManager()

            with pytest.raises(Exception, match="Clear failed"):
                manager.clear_history("session_1")


class TestChatHistoryManagerIntegration:
    """Integration tests for ChatHistoryManager."""

    def test_conversation_lifecycle(self):
        """Test complete conversation lifecycle."""
        mock_history = MagicMock()
        mock_history.messages = []

        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_history,
            ),
        ):
            manager = ChatHistoryManager()

            # Add conversation
            manager.add_to_history(
                "session_1",
                "What is federated learning?",
                "Federated learning is a distributed approach...",
            )

            # Get history
            manager.get_history("session_1")

            # Format context
            manager.format_for_context("session_1")

            # Clear history
            manager.clear_history("session_1")

            # Verify operations
            mock_history.add_messages.assert_called_once()
            mock_history.clear.assert_called_once()

    def test_multiple_sessions(self):
        """Test handling multiple sessions."""
        mock_history = MagicMock()
        mock_history.messages = [
            HumanMessage(content="User message"),
            AIMessage(content="AI response"),
        ]

        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=mock_history,
            ),
        ):
            manager = ChatHistoryManager()

            # Operate on multiple sessions
            manager.add_to_history("session_1", "User 1", "AI 1")
            manager.add_to_history("session_2", "User 2", "AI 2")
            manager.add_to_history("session_3", "User 3", "AI 3")

            history1 = manager.get_history("session_1")
            history2 = manager.get_history("session_2")
            history3 = manager.get_history("session_3")

            # All should return same mock history
            assert history1 == [("User message", "AI response")]
            assert history2 == [("User message", "AI response")]
            assert history3 == [("User message", "AI response")]

    def test_different_table_names(self):
        """Test that manager works with different table names."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.ConnectionPool",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history.PostgresChatMessageHistory",
                return_value=MagicMock(),
            ) as mock_history_class,
        ):
            manager1 = ChatHistoryManager(table_name="table_1")
            manager2 = ChatHistoryManager(table_name="table_2")

            # Create histories
            manager1._get_postgres_history("session_1")
            manager2._get_postgres_history("session_2")

            # Verify different table names used
            assert mock_history_class.call_count == 2
            assert mock_history_class.call_args_list[0][0][0] == "table_1"
            assert mock_history_class.call_args_list[1][0][0] == "table_2"
