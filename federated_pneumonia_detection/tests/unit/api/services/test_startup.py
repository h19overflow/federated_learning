import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Mock wandb before importing module under test
mock_wandb = MagicMock()
mock_wandb.__spec__ = MagicMock()
sys.modules["wandb"] = mock_wandb

from federated_pneumonia_detection.src.api.services import startup


@pytest.fixture
def mock_dependencies():
    with (
        patch(
            "federated_pneumonia_detection.src.api.services.startup.create_tables"
        ) as mock_create_tables,
        patch(
            "federated_pneumonia_detection.src.api.services.startup.start_websocket_server_thread"
        ) as mock_ws,
        patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.arxiv_mcp.MCPManager"
        ) as mock_mcp_cls,
        patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ArxivAugmentedEngine"
        ) as mock_arxiv,
        patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.rag.QueryEngine"
        ) as mock_query,
        patch(
            "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory.AgentFactory"
        ) as mock_factory,
        patch(
            "federated_pneumonia_detection.src.api.services.startup.get_wandb_tracker"
        ) as mock_wandb,
        patch(
            "federated_pneumonia_detection.src.api.services.startup.CacheProvider"
        ) as mock_cache,
        patch(
            "federated_pneumonia_detection.src.api.services.startup.MetricsService"
        ) as mock_metrics,
        patch(
            "federated_pneumonia_detection.src.api.services.startup.SummaryService"
        ) as mock_summary,
        patch(
            "federated_pneumonia_detection.src.api.services.startup.RankingService"
        ) as mock_ranking,
        patch(
            "federated_pneumonia_detection.src.api.services.startup.ExportService"
        ) as mock_export,
        patch(
            "federated_pneumonia_detection.src.api.services.startup.BackfillService"
        ) as mock_backfill,
        patch(
            "federated_pneumonia_detection.src.api.services.startup.AnalyticsFacade"
        ) as mock_facade,
        patch(
            "federated_pneumonia_detection.src.boundary.engine.dispose_engine"
        ) as mock_dispose,
    ):
        # Setup MCP Mock
        mock_mcp_instance = AsyncMock()
        mock_mcp_cls.get_instance.return_value = mock_mcp_instance

        # Setup WandB Mock
        mock_tracker = MagicMock()
        mock_wandb.return_value = mock_tracker
        mock_tracker.initialize.return_value = True

        yield {
            "create_tables": mock_create_tables,
            "ws": mock_ws,
            "mcp": mock_mcp_instance,
            "arxiv": mock_arxiv,
            "query": mock_query,
            "factory": mock_factory,
            "wandb": mock_tracker,
            "dispose": mock_dispose,
            "facade": mock_facade,
        }


@pytest.mark.asyncio
async def test_initialize_services(mock_dependencies):
    app = MagicMock()
    app.state = MagicMock()

    await startup.initialize_services(app)

    # Verify Database
    mock_dependencies["create_tables"].assert_called_once()

    # Verify WS
    mock_dependencies["ws"].assert_called_once()

    # Verify MCP
    mock_dependencies["mcp"].initialize.assert_called_once()

    # Verify Chat Services
    mock_dependencies["query"].assert_called_once()
    mock_dependencies["factory"].assert_called_once()

    # Verify Analytics
    mock_dependencies["facade"].assert_called_once()
    assert hasattr(app.state, "analytics")

    # Verify WandB
    mock_dependencies["wandb"].initialize.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_services(mock_dependencies):
    app = MagicMock()
    app.state = MagicMock()

    # Simulate initialized state
    startup._mcp_manager = mock_dependencies["mcp"]

    # Setup chat services mocks in app.state for shutdown
    mock_research_agent = MagicMock()
    mock_history_manager = MagicMock()
    mock_research_agent._history_manager = mock_history_manager

    app.state.agent_factory = MagicMock()
    app.state.agent_factory._research_agent = mock_research_agent

    mock_query_engine = MagicMock()
    mock_query_history = MagicMock()
    mock_query_engine.history_manager = mock_query_history
    app.state.query_engine = mock_query_engine

    await startup.shutdown_services(app)

    # Verify Database
    mock_dependencies["dispose"].assert_called_once()

    # Verify MCP
    mock_dependencies["mcp"].shutdown.assert_called_once()

    # Verify WandB
    mock_dependencies["wandb"].finish.assert_called_once()

    # Verify Chat Services Cleanup
    mock_history_manager.close.assert_called_once()
    mock_query_history.close.assert_called_once()
