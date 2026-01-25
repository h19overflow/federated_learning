"""
Tests for arxiv-augmented research engine.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine import (
    ArxivAugmentedEngine,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.contracts import (
    ChatInput,
)


class TestArxivAugmentedEngine:
    """Test ArxivAugmentedEngine class."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        llm = MagicMock()
        llm.ainvoke = Mock(return_value="Test response")
        llm.astream = AsyncMock()

        async def mock_astream(*args, **kwargs):
            chunks = ["Hello ", "world ", "from ", "engine"]
            for chunk in chunks:
                mock_chunk = MagicMock()
                mock_chunk.content = chunk
                yield mock_chunk

        llm.astream.side_effect = mock_astream
        return llm

    @pytest.fixture
    def mock_history_manager(self):
        """Create mock history manager."""
        manager = MagicMock()
        manager.add_to_history = Mock()
        manager.get_history = Mock(return_value=[])
        manager.clear_history = Mock()
        return manager

    @pytest.fixture
    def research_engine(self, mock_llm, mock_history_manager):
        """Create ArxivAugmentedEngine with mocked dependencies."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatGoogleGenerativeAI",
                return_value=mock_llm,
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.QueryEngine",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.create_rag_tool",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatHistoryManager",
                return_value=mock_history_manager,
            ),
        ):
            return ArxivAugmentedEngine(max_history=10)

    def test_engine_initialization(self, research_engine):
        """Test that engine initializes correctly."""
        assert research_engine._history_manager is not None
        assert research_engine.llm is not None

    def test_engine_initialization_without_rag(self, mock_llm, mock_history_manager):
        """Test engine initialization when RAG tool is unavailable."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatGoogleGenerativeAI",
                return_value=mock_llm,
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.QueryEngine",
                side_effect=Exception("RAG unavailable"),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.create_rag_tool",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatHistoryManager",
                return_value=mock_history_manager,
            ),
        ):
            engine = ArxivAugmentedEngine()
            assert engine._rag_tool is None

    def test_add_to_history(self, research_engine):
        """Test adding conversation to history."""
        research_engine.add_to_history(
            "session_1",
            "User message",
            "AI response",
        )

        research_engine._history_manager.add_to_history.assert_called_once_with(
            "session_1",
            "User message",
            "AI response",
        )

    def test_get_history(self, research_engine):
        """Test getting conversation history."""
        research_engine.get_history("session_1")

        research_engine._history_manager.get_history.assert_called_once_with(
            "session_1",
        )

    def test_clear_history(self, research_engine):
        """Test clearing conversation history."""
        research_engine.clear_history("session_1")

        research_engine._history_manager.clear_history.assert_called_once_with(
            "session_1",
        )

    @pytest.mark.asyncio
    async def test_query_stream_yields_events(self, research_engine):
        """Test that query_stream yields agent events."""
        events = []

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.stream_query",
        ) as mock_stream_query:

            async def mock_stream_gen(*args, **kwargs):
                yield {"type": "session", "session_id": "session_1"}
                yield {"type": "token", "content": "Test "}
                yield {"type": "token", "content": "response"}
                yield {"type": "done", "session_id": "session_1"}

            mock_stream_query.side_effect = mock_stream_gen

            async for event in research_engine.query_stream(
                "What is FL?",
                "session_1",
            ):
                events.append(event)

        assert len(events) == 4
        assert events[0]["type"] == "session"
        assert events[1]["type"] == "token"
        assert events[2]["type"] == "token"
        assert events[3]["type"] == "done"

    @pytest.mark.asyncio
    async def test_query_stream_with_arxiv_enabled(self, research_engine):
        """Test query_stream with arxiv enabled."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.stream_query",
        ) as mock_stream_query:

            async def mock_stream_gen(*args, **kwargs):
                yield {"type": "token", "content": "test"}

            mock_stream_query.side_effect = mock_stream_gen

            events = []
            async for event in research_engine.query_stream(
                "Search papers",
                "session_1",
                arxiv_enabled=True,
            ):
                events.append(event)

            # Verify stream_query was called with arxiv_enabled=True
            call_kwargs = mock_stream_query.call_args[1]
            assert call_kwargs["arxiv_enabled"] is True

    @pytest.mark.asyncio
    async def test_query_stream_with_original_query(self, research_engine):
        """Test query_stream with original_query parameter."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.stream_query",
        ) as mock_stream_query:

            async def mock_stream_gen(*args, **kwargs):
                yield {"type": "token", "content": "test"}

            mock_stream_query.side_effect = mock_stream_gen

            async for _ in research_engine.query_stream(
                "What is FL?",
                "session_1",
                original_query="What is federated learning?",
            ):
                pass

            call_kwargs = mock_stream_query.call_args[1]
            assert call_kwargs["original_query"] == "What is federated learning?"

    @pytest.mark.asyncio
    async def test_query_returns_dict(self, research_engine):
        """Test that query returns dict with answer."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.stream_query",
        ) as mock_stream_query:

            async def mock_stream_gen(*args, **kwargs):
                yield {"type": "token", "content": "Hello "}
                yield {"type": "token", "content": "world"}
                yield {"type": "done", "session_id": "session_1"}

            mock_stream_query.side_effect = mock_stream_gen

            result = await research_engine.query(
                ChatInput(query="What is FL?", session_id="session_1"),
            )

        assert isinstance(result, dict)
        assert "answer" in result
        assert result["answer"] == "Hello world"
        assert result["session_id"] == "session_1"

    @pytest.mark.asyncio
    async def test_query_with_tool_calls(self, research_engine):
        """Test query that includes tool calls."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.stream_query",
        ) as mock_stream_query:

            async def mock_stream_gen(*args, **kwargs):
                yield {"type": "tool_call", "tool": "rag", "args": {}}
                yield {"type": "token", "content": "Response"}

            mock_stream_query.side_effect = mock_stream_gen

            result = await research_engine.query(
                ChatInput(query="Query", session_id="session_1"),
            )

        assert "tools_used" in result
        assert len(result["tools_used"]) == 1
        assert result["tools_used"][0]["tool"] == "rag"

    @pytest.mark.asyncio
    async def test_query_with_error(self, research_engine):
        """Test query that encounters error."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.stream_query",
        ) as mock_stream_query:

            async def mock_stream_gen(*args, **kwargs):
                yield {"type": "error", "message": "Failed"}

            mock_stream_query.side_effect = mock_stream_gen

            result = await research_engine.query(
                ChatInput(query="Query", session_id="session_1"),
            )

        assert "error" in result
        assert result["error"] == "Failed"


class TestArxivAugmentedEngineErrorHandling:
    """Test error handling in ArxivAugmentedEngine."""

    @pytest.fixture
    def mock_history_manager(self):
        """Create mock history manager."""
        manager = MagicMock()
        manager.add_to_history = Mock()
        manager.get_history = Mock(return_value=[])
        manager.clear_history = Mock()
        return manager

    def test_llm_initialization_failure(self, mock_history_manager):
        """Test handling of LLM initialization failure."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatGoogleGenerativeAI",
                side_effect=Exception("LLM init failed"),
            ),
            pytest.raises(Exception, match="LLM init failed"),
        ):
            ArxivAugmentedEngine()

    def test_query_stream_propagates_errors(self, mock_history_manager):
        """Test that errors in query_stream are propagated."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatGoogleGenerativeAI",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.QueryEngine",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.create_rag_tool",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatHistoryManager",
                return_value=mock_history_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.stream_query",
                side_effect=Exception("Stream failed"),
            ),
        ):
            engine = ArxivAugmentedEngine()

            async def test_stream():
                async for _ in engine.query_stream("query", "session"):
                    pass

            import asyncio

            with pytest.raises(Exception, match="Stream failed"):
                asyncio.run(test_stream())


class TestArxivAugmentedEngineIntegration:
    """Integration tests for ArxivAugmentedEngine."""

    @pytest.fixture
    def fully_mocked_engine(self):
        """Create engine with fully mocked dependencies."""
        with (
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatGoogleGenerativeAI",
                return_value=MagicMock(),
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.QueryEngine",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.create_rag_tool",
            ),
            patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatHistoryManager",
            ),
        ):
            return ArxivAugmentedEngine(max_history=5)

    def test_engine_max_history_parameter(self, fully_mocked_engine):
        """Test that max_history parameter is respected."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ChatHistoryManager",
        ) as mock_history_manager_class:
            mock_history_manager_class.return_value = MagicMock()
            ArxivAugmentedEngine(max_history=20)

            # Verify ChatHistoryManager was called with max_history=20
            mock_history_manager_class.assert_called_once()
            call_kwargs = mock_history_manager_class.call_args[1]
            assert call_kwargs["max_history"] == 20

    def test_complete_workflow(self, fully_mocked_engine):
        """Test complete workflow with history."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.stream_query",
        ) as mock_stream_query:

            async def mock_stream_gen(*args, **kwargs):
                yield {"type": "session", "session_id": "session_1"}
                yield {"type": "token", "content": "Response"}

            mock_stream_query.side_effect = mock_stream_gen

            import asyncio

            async def run_workflow():
                # Stream response
                events = []
                async for event in fully_mocked_engine.query_stream(
                    "Query",
                    "session_1",
                ):
                    events.append(event)

                # Add to history
                fully_mocked_engine.add_to_history("session_1", "User", "AI")

                # Get history
                history = fully_mocked_engine.get_history("session_1")

                # Clear history
                fully_mocked_engine.clear_history("session_1")

                return events, history

            events, history = asyncio.run(run_workflow())

            assert len(events) > 0
            fully_mocked_engine._history_manager.add_to_history.assert_called_once()
            fully_mocked_engine._history_manager.get_history.assert_called_once()
            fully_mocked_engine._history_manager.clear_history.assert_called_once()
