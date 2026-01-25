"""
Tests for streaming utilities module.
"""

from unittest.mock import AsyncMock

import pytest

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.streaming import (
    SSEEventType,
    create_sse_event,
    execute_tool_async,
)


class TestSSEEventType:
    """Test SSEEventType enum."""

    def test_event_type_values(self):
        """Test that event types have correct string values."""
        assert SSEEventType.TOKEN.value == "token"
        assert SSEEventType.STATUS.value == "status"
        assert SSEEventType.TOOL_CALL.value == "tool_call"
        assert SSEEventType.ERROR.value == "error"
        assert SSEEventType.DONE.value == "done"

    def test_event_type_completeness(self):
        """Test that all expected event types exist."""
        event_types = [
            SSEEventType.TOKEN,
            SSEEventType.STATUS,
            SSEEventType.TOOL_CALL,
            SSEEventType.ERROR,
            SSEEventType.DONE,
        ]

        assert len(event_types) == 5


class TestCreateSSEEvent:
    """Test create_sse_event function."""

    def test_create_token_event(self):
        """Test creating token event."""
        event = create_sse_event(SSEEventType.TOKEN, content="Hello world")

        assert event["type"] == "token"
        assert event["content"] == "Hello world"
        assert len(event) == 2  # Only type and content

    def test_create_token_event_empty_content(self):
        """Test creating token event with empty content."""
        event = create_sse_event(SSEEventType.TOKEN, content="")

        assert event["type"] == "token"
        assert event["content"] == ""

    def test_create_token_event_none_content(self):
        """Test creating token event with None content (defaults to empty)."""
        event = create_sse_event(SSEEventType.TOKEN, content=None)

        assert event["type"] == "token"
        assert event["content"] == ""

    def test_create_status_event(self):
        """Test creating status event."""
        event = create_sse_event(
            SSEEventType.STATUS,
            content="Searching knowledge base...",
        )

        assert event["type"] == "status"
        assert event["content"] == "Searching knowledge base..."

    def test_create_tool_call_event(self):
        """Test creating tool_call event."""
        event = create_sse_event(
            SSEEventType.TOOL_CALL,
            tool="rag_search",
            args={"query": "federated learning"},
        )

        assert event["type"] == "tool_call"
        assert event["tool"] == "rag_search"
        assert event["args"]["query"] == "federated learning"

    def test_create_tool_call_event_no_args(self):
        """Test creating tool_call event with no args."""
        event = create_sse_event(SSEEventType.TOOL_CALL, tool="mock_tool")

        assert event["type"] == "tool_call"
        assert event["tool"] == "mock_tool"
        assert event["args"] == {}

    def test_create_error_event(self):
        """Test creating error event."""
        event = create_sse_event(
            SSEEventType.ERROR,
            message="Failed to retrieve documents",
        )

        assert event["type"] == "error"
        assert event["message"] == "Failed to retrieve documents"

    def test_create_error_event_no_message(self):
        """Test creating error event with no message (defaults to 'Unknown error')."""
        event = create_sse_event(SSEEventType.ERROR)

        assert event["type"] == "error"
        assert event["message"] == "Unknown error"

    def test_create_done_event(self):
        """Test creating done event."""
        event = create_sse_event(
            SSEEventType.DONE,
            session_id="session_123",
        )

        assert event["type"] == "done"
        assert event["session_id"] == "session_123"

    def test_create_done_event_no_session_id(self):
        """Test creating done event without session_id."""
        event = create_sse_event(SSEEventType.DONE)

        assert event["type"] == "done"
        assert "session_id" not in event

    def test_create_event_with_kwargs(self):
        """Test creating event with additional kwargs."""
        event = create_sse_event(
            SSEEventType.TOKEN,
            content="test",
            timestamp=1234567890,
            extra_field="extra_value",
        )

        assert event["type"] == "token"
        assert event["content"] == "test"
        assert event["timestamp"] == 1234567890
        assert event["extra_field"] == "extra_value"

    def test_create_all_event_types(self):
        """Test creating all event types."""
        events = [
            create_sse_event(SSEEventType.TOKEN, content="token"),
            create_sse_event(SSEEventType.STATUS, content="status"),
            create_sse_event(
                SSEEventType.TOOL_CALL,
                tool="tool",
                args={},
            ),
            create_sse_event(SSEEventType.ERROR, message="error"),
            create_sse_event(SSEEventType.DONE, session_id="session"),
        ]

        assert len(events) == 5
        assert all("type" in event for event in events)


class TestExecuteToolAsync:
    """Test execute_tool_async function."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_base_tool):
        """Test successful tool execution."""
        tools = [mock_base_tool]
        tool_name = "mock_tool"
        tool_args = {"query": "test"}

        result, error = await execute_tool_async(
            tool_name,
            tool_args,
            tools,
        )

        assert result == "Tool result"
        assert error is None
        mock_base_tool.ainvoke.assert_called_once_with(tool_args)

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mock_base_tool):
        """Test executing non-existent tool."""
        tools = [mock_base_tool]
        tool_name = "non_existent_tool"
        tool_args = {}

        result, error = await execute_tool_async(
            tool_name,
            tool_args,
            tools,
        )

        assert result is None
        assert error == "Tool 'non_existent_tool' not found"

    @pytest.mark.asyncio
    async def test_execute_tool_error(self, mock_base_tool):
        """Test tool execution with error."""
        mock_base_tool.ainvoke = AsyncMock(side_effect=Exception("Tool failed"))
        tools = [mock_base_tool]

        result, error = await execute_tool_async(
            "mock_tool",
            {},
            tools,
        )

        assert result == "Tool error: Tool failed"
        assert error == "Tool failed"

    @pytest.mark.asyncio
    async def test_execute_tool_multiple_tools(self, mock_base_tool, mock_arxiv_tool):
        """Test executing from list of multiple tools."""
        tools = [mock_base_tool, mock_arxiv_tool]

        # Execute first tool
        result1, error1 = await execute_tool_async(
            "mock_tool",
            {"query": "test"},
            tools,
        )
        assert result1 == "Tool result"
        assert error1 is None

        # Execute second tool
        result2, error2 = await execute_tool_async(
            "arxiv_search",
            {"query": "papers"},
            tools,
        )
        assert result2 == "Found 5 papers on federated learning"
        assert error2 is None

    @pytest.mark.asyncio
    async def test_execute_tool_with_complex_args(self, mock_base_tool):
        """Test tool execution with complex arguments."""
        tools = [mock_base_tool]
        tool_args = {
            "query": "federated learning",
            "max_results": 10,
            "filters": {"year": 2024},
        }

        result, error = await execute_tool_async(
            "mock_tool",
            tool_args,
            tools,
        )

        assert result == "Tool result"
        assert error is None
        mock_base_tool.ainvoke.assert_called_once_with(tool_args)

    @pytest.mark.asyncio
    async def test_execute_tool_empty_tools_list(self):
        """Test executing tool with empty tools list."""
        tools = []

        result, error = await execute_tool_async(
            "any_tool",
            {},
            tools,
        )

        assert result is None
        assert "not found" in error

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self, mock_base_tool):
        """Test tool execution with timeout."""
        mock_base_tool.ainvoke = AsyncMock(side_effect=TimeoutError("Timeout"))
        tools = [mock_base_tool]

        result, error = await execute_tool_async(
            "mock_tool",
            {},
            tools,
        )

        assert result == "Tool error: Timeout"
        assert error == "Timeout"


class TestStreamingIntegration:
    """Integration tests for streaming utilities."""

    def test_sse_event_serialization(self):
        """Test that SSE events can be serialized to JSON."""
        import json

        event = create_sse_event(
            SSEEventType.TOKEN,
            content="test",
            extra="data",
        )

        # Should be JSON serializable
        json_str = json.dumps(event)
        assert "test" in json_str
        assert "token" in json_str

    def test_sse_event_types_are_strings(self):
        """Test that all event types are strings."""
        events = [
            create_sse_event(SSEEventType.TOKEN, content="t"),
            create_sse_event(SSEEventType.STATUS, content="s"),
            create_sse_event(SSEEventType.TOOL_CALL, tool="t", args={}),
            create_sse_event(SSEEventType.ERROR, message="e"),
            create_sse_event(SSEEventType.DONE, session_id="s"),
        ]

        for event in events:
            assert isinstance(event["type"], str)

    @pytest.mark.asyncio
    async def test_execute_tool_chain(self, mock_base_tool, mock_arxiv_tool):
        """Test executing multiple tools in sequence."""
        tools = [mock_base_tool, mock_arxiv_tool]

        # Execute chain
        result1, _ = await execute_tool_async(
            "mock_tool",
            {"query": "1"},
            tools,
        )
        result2, _ = await execute_tool_async(
            "arxiv_search",
            {"query": "2"},
            tools,
        )

        assert result1 == "Tool result"
        assert result2 == "Found 5 papers on federated learning"
