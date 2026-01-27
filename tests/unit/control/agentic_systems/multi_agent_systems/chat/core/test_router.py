"""
Tests for query router module.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core import (  # noqa: E501
    router,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router import (  # noqa: E501
    ROUTER_CLASSIFICATION_PROMPT,
    QueryClassification,
    _get_router_llm,
    classify_query,
)


@pytest.fixture(autouse=True)
def reset_router_llm():
    """Reset the router LLM singleton before each test."""
    router._router_llm = None
    yield
    router._router_llm = None


class TestQueryClassification:
    """Test QueryClassification Pydantic model."""

    def test_query_classification_model(self):
        """Test that QueryClassification can be created."""
        classification = QueryClassification(mode="research")

        assert classification.mode == "research"

    def test_query_classification_basic(self):
        """Test QueryClassification with basic mode."""
        classification = QueryClassification(mode="basic")

        assert classification.mode == "basic"

    def test_query_classification_invalid_mode(self):
        """Test QueryClassification with invalid mode."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            QueryClassification(mode="invalid")


class TestGetRouterLLM:
    """Test _get_router_llm function."""

    def test_get_router_llm_creates_llm(self):
        """Test that _get_router_llm creates LLM instance."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router.ChatGoogleGenerativeAI",
        ) as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_instance.with_structured_output = Mock(
                return_value=mock_llm_instance,
            )
            mock_llm_class.return_value = mock_llm_instance

            llm = _get_router_llm()

            assert llm is not None
            mock_llm_class.assert_called_once_with(
                model="gemini-2.0-flash-exp",
                temperature=0.0,
                max_tokens=50,
            )
            mock_llm_instance.with_structured_output.assert_called_once_with(
                QueryClassification,
            )

    def test_get_router_llm_caches_instance(self):
        """Test that _get_router_llm caches the LLM instance."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router.ChatGoogleGenerativeAI",
        ) as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_instance.with_structured_output = Mock(
                return_value=mock_llm_instance,
            )
            mock_llm_class.return_value = mock_llm_instance

            # Call multiple times
            llm1 = _get_router_llm()
            llm2 = _get_router_llm()
            llm3 = _get_router_llm()

            # Should only create once
            mock_llm_class.assert_called_once()
            assert llm1 is llm2 is llm3

    def test_get_router_llm_with_structured_output(self):
        """Test that _get_router_llm applies structured output."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router.ChatGoogleGenerativeAI",
        ) as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_instance.with_structured_output = Mock(
                return_value=mock_llm_instance,
            )
            mock_llm_class.return_value = mock_llm_instance

            _get_router_llm()

            # Verify with_structured_output was called with correct model
            mock_llm_instance.with_structured_output.assert_called_once()
            call_args = mock_llm_instance.with_structured_output.call_args[0][0]
            assert call_args == QueryClassification


class TestClassifyQuery:
    """Test classify_query function."""

    def test_classify_query_research_mode(self):
        """Test classifying a research query."""
        mock_response = MagicMock()
        mock_response.mode = "research"

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            mode = classify_query("What papers discuss federated learning?")

            assert mode == "research"
            mock_llm.invoke.assert_called_once()

    def test_classify_query_basic_mode(self):
        """Test classifying a basic query."""
        mock_response = MagicMock()
        mock_response.mode = "basic"

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            mode = classify_query("Thanks for the explanation!")

            assert mode == "basic"
            mock_llm.invoke.assert_called_once()

    def test_classify_query_with_arxiv(self):
        """Test classifying a query that needs arxiv search."""
        mock_response = MagicMock()
        mock_response.mode = "research"

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            mode = classify_query("Find papers by Andrew Ng")

            assert mode == "research"

    def test_classify_query_with_comparison(self):
        """Test classifying a comparison query."""
        mock_response = MagicMock()
        mock_response.mode = "research"

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            mode = classify_query("Compare ResNet and VGG architectures")

            assert mode == "research"

    def test_classify_query_greeting(self):
        """Test classifying a greeting."""
        mock_response = MagicMock()
        mock_response.mode = "basic"

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            mode = classify_query("Hello!")

            assert mode == "basic"

    def test_classify_query_uses_prompt_template(self):
        """Test that classify_query uses the prompt template."""
        mock_response = MagicMock()
        mock_response.mode = "research"

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            query = "Test query"
            classify_query(query)

            # Verify the prompt was formatted correctly
            call_args = mock_llm.invoke.call_args[0][0]
            assert query in call_args
            assert "research" in call_args
            assert "basic" in call_args


class TestClassifyQueryErrorHandling:
    """Test error handling in classify_query."""

    def test_classify_query_llm_failure_defaults_to_research(self):
        """Test that LLM failure defaults to research mode (safe fallback)."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(side_effect=Exception("API Error"))
            mock_get_llm.return_value = mock_llm

            mode = classify_query("Test query")

            # Should default to "research" on error
            assert mode == "research"

    def test_classify_query_get_llm_failure(self):
        """Test that get_llm failure defaults to research mode."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
            side_effect=Exception("Failed to initialize LLM"),
        ):
            mode = classify_query("Test query")

            # Should default to "research" on error
            assert mode == "research"

    def test_classify_query_timeout(self):
        """Test handling of timeout errors."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(side_effect=TimeoutError("Request timed out"))
            mock_get_llm.return_value = mock_llm

            mode = classify_query("Test query")

            # Should default to "research" on timeout
            assert mode == "research"


class TestRouterPrompt:
    """Test router prompt template."""

    def test_router_prompt_contains_keywords(self):
        """Test that router prompt contains expected keywords."""
        prompt = ROUTER_CLASSIFICATION_PROMPT

        assert "research" in prompt
        assert "basic" in prompt
        assert "papers" in prompt
        assert "comparisons" in prompt
        assert "citations" in prompt

    def test_router_prompt_research_examples(self):
        """Test that prompt has research examples."""
        prompt = ROUTER_CLASSIFICATION_PROMPT

        assert "Questions about papers" in prompt
        assert "Requests for comparisons" in prompt
        assert "technical details" in prompt

    def test_router_prompt_basic_examples(self):
        """Test that prompt has basic examples."""
        prompt = ROUTER_CLASSIFICATION_PROMPT

        assert "Greetings" in prompt
        assert "thanks" in prompt
        assert "acknowledgments" in prompt
        assert "Clarifications" in prompt

    def test_router_prompt_formatting(self):
        """Test that router prompt can be formatted."""
        query = "What is machine learning?"

        formatted_prompt = ROUTER_CLASSIFICATION_PROMPT.format(query=query)

        assert query in formatted_prompt
        assert "{query}" not in formatted_prompt


class TestRouterIntegration:
    """Integration tests for query router."""

    def test_multiple_queries_classification(self):
        """Test classifying multiple different queries."""
        mock_response = MagicMock()

        queries_and_modes = [
            ("What is federated learning?", "research"),
            ("Hello!", "basic"),
            ("Find papers on deep learning", "research"),
            ("Thanks!", "basic"),
            ("Compare CNN and RNN", "research"),
            ("That was helpful", "basic"),
        ]

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(side_effect=lambda x: mock_response)
            mock_get_llm.return_value = mock_llm

            for query, expected_mode in queries_and_modes:
                mock_response.mode = expected_mode
                mode = classify_query(query)
                assert mode == expected_mode

            # Verify LLM was called for each query
            assert mock_llm.invoke.call_count == len(queries_and_modes)

    def test_router_with_long_query(self):
        """Test router with very long query."""
        long_query = "What is federated learning? " * 100

        mock_response = MagicMock()
        mock_response.mode = "research"

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = Mock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            mode = classify_query(long_query)

            assert mode == "research"
            # Should truncate in logs but still send full query to LLM
            mock_llm.invoke.assert_called_once()
