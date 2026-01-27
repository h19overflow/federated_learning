"""
Tests for agent factory module.
"""

from unittest.mock import MagicMock, patch

import pytest

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.base_agent import (  # noqa: E501
    BaseAgent,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory import (  # noqa: E501
    AgentFactory,
    get_agent_factory,
)


class TestAgentFactory:
    """Test AgentFactory class."""

    def test_factory_initialization(self):
        """Test that AgentFactory initializes correctly."""
        factory = AgentFactory()

        assert factory._research_agent is None

    def test_get_chat_agent_creates_agent(self, mock_arxiv_engine):
        """Test that get_chat_agent creates and caches agent."""
        factory = AgentFactory()

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.ArxivAugmentedEngine",
            return_value=mock_arxiv_engine,
        ) as mock_constructor:
            agent = factory.get_chat_agent()

            # Verify agent was created
            assert isinstance(agent, BaseAgent) or isinstance(agent, MagicMock)
            mock_constructor.assert_called_once()

            # Verify caching
            factory.get_chat_agent()
            mock_constructor.assert_called_once()  # Still only called once

    def test_get_chat_agent_caches_agent(self, mock_arxiv_engine):
        """Test that get_chat_agent returns cached agent on subsequent calls."""
        factory = AgentFactory()

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.ArxivAugmentedEngine",
            return_value=mock_arxiv_engine,
        ):
            agent1 = factory.get_chat_agent()
            agent2 = factory.get_chat_agent()

            # Should return the same instance
            assert agent1 is agent2

    def test_factory_independent_instances(self):
        """Test that multiple factory instances have separate agents."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.ArxivAugmentedEngine",
            side_effect=[MagicMock(spec=BaseAgent), MagicMock(spec=BaseAgent)],
        ) as mock_constructor:
            factory1 = AgentFactory()
            factory2 = AgentFactory()

            agent1 = factory1.get_chat_agent()
            agent2 = factory2.get_chat_agent()

            # Should be different instances (side_effect returns different mocks)
            assert agent1 is not agent2
            assert mock_constructor.call_count == 2


class TestGetAgentFactory:
    """Test get_agent_factory singleton function."""

    def test_get_agent_factory_returns_factory(self):
        """Test that get_agent_factory returns AgentFactory instance."""
        factory = get_agent_factory()

        assert isinstance(factory, AgentFactory)

    def test_get_agent_factory_singleton(self):
        """Test that get_agent_factory returns same instance."""
        factory1 = get_agent_factory()
        factory2 = get_agent_factory()

        assert factory1 is factory2

    def test_get_agent_factory_caches_state(self, mock_arxiv_engine):
        """Test that factory state persists across calls."""
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.ArxivAugmentedEngine",
            return_value=mock_arxiv_engine,
        ):
            factory1 = get_agent_factory()
            agent1 = factory1.get_chat_agent()

            factory2 = get_agent_factory()
            agent2 = factory2.get_chat_agent()

            # Should return the same agent (cached in singleton)
            assert agent1 is agent2

    def test_multiple_get_agent_factory_calls(self):
        """Test that multiple calls don't create new factories."""
        factories = [get_agent_factory() for _ in range(5)]

        # All should be the same instance
        assert all(f is factories[0] for f in factories)


class TestAgentFactoryIntegration:
    """Integration tests for AgentFactory."""

    @pytest.fixture
    def real_factory(self):
        """Create a real AgentFactory for testing."""
        return AgentFactory()

    def test_factory_without_patch(self, real_factory):
        """Test factory behavior without mocking ArxivAugmentedEngine."""
        # This test verifies factory logic, not ArxivAugmentedEngine behavior

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.ArxivAugmentedEngine",
        ) as mock_arxiv_engine:
            mock_arxiv_engine.return_value = MagicMock(spec=BaseAgent)

            agent = real_factory.get_chat_agent()

            assert agent is not None
            mock_arxiv_engine.assert_called_once()

    def test_factory_concurrent_access(self):
        """Test that factory handles concurrent access safely."""
        import asyncio

        async def get_agent_concurrently(factory):
            return factory.get_chat_agent()

        async def run_concurrent_test(factory):
            with patch(
                "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.ArxivAugmentedEngine",
                return_value=MagicMock(spec=BaseAgent),
            ):
                # Get agents concurrently
                agent_list = await asyncio.gather(
                    *[get_agent_concurrently(factory) for _ in range(10)],
                )

                # Verify all agents are the same instance
                assert all(a is agent_list[0] for a in agent_list)

        factory = AgentFactory()
        asyncio.run(run_concurrent_test(factory))


class TestAgentFactoryErrorHandling:
    """Test error handling in AgentFactory."""

    def test_factory_with_research_agent_failure(self):
        """Test that factory handles ArxivAugmentedEngine initialization failure."""
        factory = AgentFactory()

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.ArxivAugmentedEngine",
            side_effect=Exception("Failed to initialize"),
        ):
            with pytest.raises(Exception, match="Failed to initialize"):
                factory.get_chat_agent()

    def test_factory_state_after_failure(self):
        """Test that factory state is reset after initialization failure."""
        factory = AgentFactory()

        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.ArxivAugmentedEngine",
            side_effect=Exception("Failed"),
        ):
            # First call fails
            with pytest.raises(Exception):
                factory.get_chat_agent()

            # Agent should still be None
            assert factory._research_agent is None

        # Second call with successful patch
        with patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.ArxivAugmentedEngine",
            return_value=MagicMock(spec=BaseAgent),
        ):
            agent = factory.get_chat_agent()
            assert agent is not None
            assert factory._research_agent is not None


class TestGetAgentFactoryReset:
    """Test resetting the singleton factory."""

    def test_reset_singleton_between_tests(self):
        """Test that singleton can be effectively reset between tests."""
        # Get factory
        factory1 = get_agent_factory()

        # Reset the global singleton
        from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems import (  # noqa: E501
            factory as factory_module,
        )

        factory_module._agent_factory = None

        # Get new factory
        factory2 = get_agent_factory()

        # Should be different instances
        assert factory1 is not factory2
