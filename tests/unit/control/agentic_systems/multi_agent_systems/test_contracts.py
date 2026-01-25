import pytest
from pydantic import TypeAdapter, ValidationError
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.contracts import (
    ChatInput,
    AgentEvent,
)


def test_chat_input_validation():
    """Test ChatInput validation using Pydantic TypeAdapter."""
    adapter = TypeAdapter(ChatInput)

    # Valid input
    valid_data = {"query": "Hello", "session_id": "session-123", "arxiv_enabled": True}
    chat_input = adapter.validate_python(valid_data)
    assert chat_input.query == "Hello"
    assert chat_input.session_id == "session-123"
    assert chat_input.arxiv_enabled is True

    # Missing required field
    with pytest.raises(ValidationError):
        adapter.validate_python({"query": "Hello"})  # session_id missing

    # Invalid type
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "query": "Hello",
                "session_id": 123,  # should be str
                "arxiv_enabled": "not-a-bool",
            }
        )


def test_agent_event_validation():
    """Test AgentEvent validation using Pydantic TypeAdapter."""
    adapter = TypeAdapter(AgentEvent)

    # Valid input
    valid_data = {
        "type": "token",
        "content": "Hello world",
        "session_id": "session-123",
    }
    event = adapter.validate_python(valid_data)
    assert event["type"] == "token"
    assert event["content"] == "Hello world"

    # TypedDict with total=False allows missing fields
    minimal_data = {"type": "done"}
    event = adapter.validate_python(minimal_data)
    assert event["type"] == "done"

    # Invalid type
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "type": 123,  # should be str
            }
        )
