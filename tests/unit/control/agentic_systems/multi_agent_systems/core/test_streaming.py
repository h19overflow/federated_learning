from unittest.mock import MagicMock

import pytest

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.stream_research import (  # noqa: E501
    stream_research,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.stream_state import (  # noqa: E501
    StreamState,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.streaming import (  # noqa: E501
    SSEEventType,
)


@pytest.mark.asyncio
async def test_process_stream_tokens():
    """
    Test streaming tokens and accumulating final answer.
    Target: stream_research (StreamingOrchestrator equivalent)
    """
    agent = MagicMock()

    # Mock tokens mimicking LangGraph/LangChain message chunks
    # We use a class to avoid MagicMock's default truthy attributes
    class MockToken:
        def __init__(self, content=None, text=None, tool_call_chunks=None):
            self.content = content
            self.text = text
            self.tool_call_chunks = tool_call_chunks or []

    token1 = MockToken(content="Federated")
    token2 = MockToken(content=" Learning")

    metadata = {"langgraph_node": "agent"}

    async def mock_astream(*args, **kwargs):
        yield token1, metadata
        yield token2, metadata

    agent.astream = mock_astream

    state = StreamState()
    events = []
    async for event in stream_research(agent, [], state):
        events.append(event)

    # Verify final answer accumulation (Rule 1: Assert logic)
    assert state.full_response == "Federated Learning"

    # Verify events (SSE format)
    assert len(events) == 2
    assert events[0]["type"] == SSEEventType.TOKEN.value
    assert events[0]["content"] == "Federated"
    assert events[1]["type"] == SSEEventType.TOKEN.value
    assert events[1]["content"] == " Learning"


@pytest.mark.asyncio
async def test_process_stream_tool_calls():
    """
    Test streaming tool calls.
    Target: stream_research
    """
    agent = MagicMock()

    # Tool call chunk mimicking LangGraph event
    class MockToken:
        def __init__(self, content=None, text=None, tool_call_chunks=None):
            self.content = content
            self.text = text
            self.tool_call_chunks = tool_call_chunks or []

    token_tool = MockToken(
        tool_call_chunks=[{"name": "arxiv_search", "args": {"query": "pneumonia"}}]
    )
    token_text = MockToken(content="Found some papers.")

    metadata = {"langgraph_node": "agent"}

    async def mock_astream(*args, **kwargs):
        yield token_tool, metadata
        yield token_text, metadata

    agent.astream = mock_astream

    state = StreamState()
    events = []
    async for event in stream_research(agent, [], state):
        events.append(event)

    # stream_research yields STATUS then TOOL_CALL for new tools, then TOKEN
    assert len(events) == 3
    assert events[0]["type"] == SSEEventType.STATUS.value
    assert "Arxiv Search" in events[0]["content"]
    assert events[1]["type"] == SSEEventType.TOOL_CALL.value
    assert events[1]["tool"] == "arxiv_search"
    assert events[2]["type"] == SSEEventType.TOKEN.value
    assert events[2]["content"] == "Found some papers."

    # Verify state tracking
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0]["name"] == "arxiv_search"
    assert state.full_response == "Found some papers."


@pytest.mark.asyncio
async def test_process_stream_error():
    """
    Test error handling during stream.
    """
    agent = MagicMock()

    # Mock a failure during iteration
    async def mock_astream_fail(*args, **kwargs):
        yield MagicMock(), {"langgraph_node": "agent"}
        raise Exception("Agent crashed")

    agent.astream = mock_astream_fail

    state = StreamState()
    events = []
    async for event in stream_research(agent, [], state):
        events.append(event)

    # Should yield at least one event (token/status) then an ERROR event
    assert any(e["type"] == SSEEventType.ERROR.value for e in events)
    error_event = next(e for e in events if e["type"] == SSEEventType.ERROR.value)
    assert "Agent execution failed" in error_event["message"]
