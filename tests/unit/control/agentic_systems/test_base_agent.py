"""
Tests for base_agent module (abstract base class for agents).
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.base_agent import (
    BaseAgent,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.contracts import (
    ChatInput,
)


class TestBaseAgent:
    """Test BaseAgent abstract base class."""

    def test_base_agent_is_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore

    def test_base_agent_subclass_with_stream(self):
        """Test creating concrete subclass with stream method."""

        class ConcreteAgent(BaseAgent):
            async def stream(self, chat_input):
                yield {"type": "token", "content": "test"}

            async def query(self, chat_input):
                return {"answer": "test"}

            def history(self, session_id):
                return []

            def clear_history(self, session_id):
                pass

        agent = ConcreteAgent()
        assert isinstance(agent, BaseAgent)

    def test_base_agent_subclass_with_all_methods(self):
        """Test creating concrete subclass with all required methods."""

        class ConcreteAgent(BaseAgent):
            async def stream(self, chat_input):
                yield {"type": "token", "content": "test"}
                yield {"type": "done", "session_id": chat_input.session_id}

            async def query(self, chat_input):
                return {"answer": "response", "session_id": chat_input.session_id}

            def history(self, session_id):
                return [("user_msg", "ai_response")]

            def clear_history(self, session_id):
                pass

        agent = ConcreteAgent()
        assert isinstance(agent, BaseAgent)


class TestBaseAgentConcreteImplementation:
    """Test a concrete implementation of BaseAgent."""

    @pytest.fixture
    def concrete_agent(self):
        """Create a concrete agent for testing."""

        class MockAgent(BaseAgent):
            def __init__(self):
                self._history = {}

            async def stream(self, chat_input):
                yield {"type": "session", "session_id": chat_input.session_id}
                yield {"type": "status", "content": "Processing..."}
                yield {"type": "token", "content": "Hello "}
                yield {"type": "token", "content": "world"}
                yield {"type": "done", "session_id": chat_input.session_id}

            async def query(self, chat_input):
                return {
                    "answer": "Hello world",
                    "session_id": chat_input.session_id,
                }

            def history(self, session_id):
                return self._history.get(session_id, [])

            def clear_history(self, session_id):
                self._history[session_id] = []

        return MockAgent()

    @pytest.mark.asyncio
    async def test_stream_yields_events(self, concrete_agent, sample_chat_input):
        """Test that stream method yields correct events."""
        events = []
        async for event in concrete_agent.stream(sample_chat_input):
            events.append(event)

        assert len(events) == 5
        assert events[0]["type"] == "session"
        assert events[0]["session_id"] == sample_chat_input.session_id
        assert events[1]["type"] == "status"
        assert events[2]["type"] == "token"
        assert events[3]["type"] == "token"
        assert events[4]["type"] == "done"

    @pytest.mark.asyncio
    async def test_stream_with_different_sessions(self, concrete_agent):
        """Test stream with different session IDs."""
        input1 = ChatInput(query="test1", session_id="session_1")
        input2 = ChatInput(query="test2", session_id="session_2")

        events1 = [e async for e in concrete_agent.stream(input1)]
        events2 = [e async for e in concrete_agent.stream(input2)]

        assert events1[0]["session_id"] == "session_1"
        assert events2[0]["session_id"] == "session_2"

    @pytest.mark.asyncio
    async def test_query_returns_dict(self, concrete_agent, sample_chat_input):
        """Test that query returns a dictionary."""
        result = await concrete_agent.query(sample_chat_input)

        assert isinstance(result, dict)
        assert "answer" in result
        assert "session_id" in result
        assert result["session_id"] == sample_chat_input.session_id

    @pytest.mark.asyncio
    async def test_query_matches_stream_content(
        self,
        concrete_agent,
        sample_chat_input,
    ):
        """Test that query result matches streamed content."""
        result = await concrete_agent.query(sample_chat_input)

        events = []
        async for event in concrete_agent.stream(sample_chat_input):
            if event["type"] == "token":
                events.append(event["content"])

        streamed_content = "".join(events)
        assert result["answer"] == streamed_content

    def test_history_returns_list_of_tuples(self, concrete_agent):
        """Test that history returns list of tuples."""
        history = concrete_agent.history("session_1")

        assert isinstance(history, list)
        # Empty history should return empty list
        assert len(history) == 0

    def test_clear_history_clears_session(self, concrete_agent):
        """Test that clear_history clears session history."""
        session_id = "session_1"

        # Add some history
        concrete_agent._history[session_id] = [
            ("user_msg", "ai_response"),
            ("another_msg", "another_response"),
        ]

        # Verify history exists
        assert len(concrete_agent.history(session_id)) == 2

        # Clear history
        concrete_agent.clear_history(session_id)

        # Verify history is cleared
        assert len(concrete_agent.history(session_id)) == 0


class TestBaseAgentErrorHandling:
    """Test error handling in BaseAgent implementations."""

    @pytest.fixture
    def failing_agent(self):
        """Create agent that raises errors."""

        class FailingAgent(BaseAgent):
            async def stream(self, chat_input):
                yield {"type": "session", "session_id": chat_input.session_id}
                yield {"type": "error", "message": "Test error"}
                yield {"type": "done", "session_id": chat_input.session_id}

            async def query(self, chat_input):
                raise ValueError("Query failed")

            def history(self, session_id):
                raise RuntimeError("History access failed")

            def clear_history(self, session_id):
                raise RuntimeError("Clear failed")

        return FailingAgent()

    @pytest.mark.asyncio
    async def test_stream_with_error_event(self, failing_agent, sample_chat_input):
        """Test that stream can yield error events."""
        events = []
        async for event in failing_agent.stream(sample_chat_input):
            events.append(event)

        assert len(events) == 3
        assert events[1]["type"] == "error"
        assert events[1]["message"] == "Test error"

    @pytest.mark.asyncio
    async def test_query_raises_exception(self, failing_agent, sample_chat_input):
        """Test that query can raise exceptions."""
        with pytest.raises(ValueError, match="Query failed"):
            await failing_agent.query(sample_chat_input)

    def test_history_raises_exception(self, failing_agent):
        """Test that history can raise exceptions."""
        with pytest.raises(RuntimeError, match="History access failed"):
            failing_agent.history("session_1")

    def test_clear_history_raises_exception(self, failing_agent):
        """Test that clear_history can raise exceptions."""
        with pytest.raises(RuntimeError, match="Clear failed"):
            failing_agent.clear_history("session_1")


class TestBaseAgentWithMock:
    """Test BaseAgent using mock implementations."""

    @pytest.fixture
    def mock_base_agent(self):
        """Create a fully mocked BaseAgent."""
        agent = MagicMock(spec=BaseAgent)

        # Mock async stream
        async def mock_stream(chat_input):
            yield {"type": "session", "session_id": chat_input.session_id}
            yield {"type": "token", "content": "mock"}
            yield {"type": "done", "session_id": chat_input.session_id}

        agent.stream = MagicMock(side_effect=mock_stream)

        agent.query = AsyncMock(return_value={"answer": "mock", "session_id": "test"})
        agent.history = Mock(return_value=[("user", "ai")])
        agent.clear_history = Mock()

        return agent

    @pytest.mark.asyncio
    async def test_mock_agent_stream(self, mock_base_agent, sample_chat_input):
        """Test that mocked agent streams correctly."""
        events = []
        async for event in mock_base_agent.stream(sample_chat_input):
            events.append(event)

        assert len(events) == 3
        mock_base_agent.stream.assert_called_once_with(sample_chat_input)

    @pytest.mark.asyncio
    async def test_mock_agent_query(self, mock_base_agent, sample_chat_input):
        """Test that mocked agent queries correctly."""
        result = await mock_base_agent.query(sample_chat_input)

        assert result["answer"] == "mock"
        mock_base_agent.query.assert_called_once_with(sample_chat_input)

    def test_mock_agent_history(self, mock_base_agent):
        """Test that mocked agent returns history."""
        history = mock_base_agent.history("session_1")

        assert history == [("user", "ai")]
        mock_base_agent.history.assert_called_once_with("session_1")

    def test_mock_agent_clear_history(self, mock_base_agent):
        """Test that mocked agent clears history."""
        mock_base_agent.clear_history("session_1")

        mock_base_agent.clear_history.assert_called_once_with("session_1")
