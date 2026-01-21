"""
Tests for contracts module (shared types and enums).
"""

import pytest

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.contracts import (
    AgentEvent,
    AgentEventType,
    ChatInput,
)


class TestAgentEventType:
    """Test AgentEventType enum."""

    def test_event_type_values(self):
        """Test that event types have correct string values."""
        assert AgentEventType.SESSION.value == "session"
        assert AgentEventType.TOKEN.value == "token"
        assert AgentEventType.STATUS.value == "status"
        assert AgentEventType.TOOL_CALL.value == "tool_call"
        assert AgentEventType.ERROR.value == "error"
        assert AgentEventType.DONE.value == "done"

    def test_event_type_enumeration(self):
        """Test that all event types are accessible."""
        event_types = [
            AgentEventType.SESSION,
            AgentEventType.TOKEN,
            AgentEventType.STATUS,
            AgentEventType.TOOL_CALL,
            AgentEventType.ERROR,
            AgentEventType.DONE,
        ]

        assert len(event_types) == 6


class TestChatInput:
    """Test ChatInput dataclass."""

    def test_chat_input_creation(self):
        """Test creating ChatInput with required fields."""
        chat_input = ChatInput(
            query="What is federated learning?",
            session_id="test_session_123",
        )

        assert chat_input.query == "What is federated learning?"
        assert chat_input.session_id == "test_session_123"
        assert chat_input.arxiv_enabled is False  # Default value
        assert chat_input.run_id is None  # Default value
        assert chat_input.original_query is None  # Default value

    def test_chat_input_with_all_fields(self):
        """Test creating ChatInput with all optional fields."""
        chat_input = ChatInput(
            query="Search for papers",
            session_id="session_456",
            arxiv_enabled=True,
            run_id=42,
            original_query="Papers about federated learning",
        )

        assert chat_input.query == "Search for papers"
        assert chat_input.session_id == "session_456"
        assert chat_input.arxiv_enabled is True
        assert chat_input.run_id == 42
        assert chat_input.original_query == "Papers about federated learning"

    def test_chat_input_immutability(self):
        """Test that ChatInput is frozen (immutable)."""
        chat_input = ChatInput(query="test", session_id="session_1")

        # Frozen dataclass should not allow modification
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            chat_input.query = "modified"

    def test_chat_input_defaults(self):
        """Test that default values are applied correctly."""
        chat_input = ChatInput(query="test", session_id="session_1")

        assert chat_input.arxiv_enabled is False
        assert chat_input.run_id is None
        assert chat_input.original_query is None


class TestAgentEvent:
    """Test AgentEvent TypedDict."""

    def test_agent_event_token(self):
        """Test creating token event."""
        event: AgentEvent = {
            "type": "token",
            "content": "Hello ",
        }

        assert event["type"] == "token"
        assert event["content"] == "Hello "

    def test_agent_event_status(self):
        """Test creating status event."""
        event: AgentEvent = {
            "type": "status",
            "content": "Searching database...",
        }

        assert event["type"] == "status"
        assert event["content"] == "Searching database..."

    def test_agent_event_tool_call(self):
        """Test creating tool_call event."""
        event: AgentEvent = {
            "type": "tool_call",
            "tool": "rag_search",
            "args": {"query": "federated learning"},
        }

        assert event["type"] == "tool_call"
        assert event["tool"] == "rag_search"
        assert event["args"]["query"] == "federated learning"

    def test_agent_event_error(self):
        """Test creating error event."""
        event: AgentEvent = {
            "type": "error",
            "message": "Failed to retrieve documents",
        }

        assert event["type"] == "error"
        assert event["message"] == "Failed to retrieve documents"

    def test_agent_event_session(self):
        """Test creating session event."""
        event: AgentEvent = {
            "type": "session",
            "session_id": "session_123",
        }

        assert event["type"] == "session"
        assert event["session_id"] == "session_123"

    def test_agent_event_done(self):
        """Test creating done event."""
        event: AgentEvent = {
            "type": "done",
            "session_id": "session_123",
        }

        assert event["type"] == "done"
        assert event["session_id"] == "session_123"

    def test_agent_event_with_additional_fields(self):
        """Test that AgentEvent accepts additional fields."""
        event: AgentEvent = {
            "type": "token",
            "content": "test",
            "timestamp": 1234567890,
            "metadata": {"extra": "data"},
        }

        assert event["type"] == "token"
        assert event["content"] == "test"
        assert event["timestamp"] == 1234567890
        assert event["metadata"]["extra"] == "data"


class TestContractsIntegration:
    """Integration tests for contracts."""

    def test_chat_input_and_event_flow(self):
        """Test flow from ChatInput to AgentEvent."""
        chat_input = ChatInput(
            query="What is federated learning?",
            session_id="session_1",
        )

        # Simulate events that would be generated
        events: list[AgentEvent] = [
            {"type": "session", "session_id": chat_input.session_id},
            {"type": "token", "content": chat_input.query[:10]},
            {"type": "done", "session_id": chat_input.session_id},
        ]

        assert len(events) == 3
        assert events[0]["session_id"] == chat_input.session_id
        assert events[2]["session_id"] == chat_input.session_id

    def test_event_type_enum_matches_event_type_string(self):
        """Test that enum values match event type strings."""
        events_to_test = [
            (AgentEventType.SESSION, "session"),
            (AgentEventType.TOKEN, "token"),
            (AgentEventType.STATUS, "status"),
            (AgentEventType.TOOL_CALL, "tool_call"),
            (AgentEventType.ERROR, "error"),
            (AgentEventType.DONE, "done"),
        ]

        for event_type, expected_value in events_to_test:
            assert event_type.value == expected_value
            event: AgentEvent = {"type": expected_value}
            assert event["type"] == expected_value
