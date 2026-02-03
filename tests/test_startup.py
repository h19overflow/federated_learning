import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from federated_pneumonia_detection.src.api.services.startup import (
    initialize_services,
    shutdown_services,
)

@pytest.fixture
def mock_app():
    app = MagicMock()
    app.state = MagicMock()
    return app

@pytest.mark.asyncio
async def test_initialize_services_success(mock_app):
    """Test successful initialization of all services."""
    with patch("federated_pneumonia_detection.src.api.services.startup._init_db") as mock_init_db, \
         patch("federated_pneumonia_detection.src.api.services.startup._init_ws") as mock_init_ws, \
         patch("federated_pneumonia_detection.src.api.services.startup._init_mcp", new_callable=AsyncMock) as mock_init_mcp, \
         patch("federated_pneumonia_detection.src.api.services.startup._init_chat", new_callable=AsyncMock) as mock_init_chat, \
         patch("federated_pneumonia_detection.src.api.services.startup._init_analytics") as mock_init_analytics, \
         patch("federated_pneumonia_detection.src.api.services.startup._init_wandb") as mock_init_wandb:
        
        mock_init_chat.return_value = (MagicMock(), MagicMock())
        
        await initialize_services(mock_app)
        
        mock_init_db.assert_called_once()
        mock_init_ws.assert_called_once()
        mock_init_mcp.assert_called_once()
        mock_init_chat.assert_called_once_with(mock_app)
        mock_init_analytics.assert_called_once_with(mock_app)
        mock_init_wandb.assert_called_once()

@pytest.mark.asyncio
async def test_initialize_services_db_failure():
    """Test that database failure raises an exception and stops initialization."""
    with patch("federated_pneumonia_detection.src.api.services.startup._init_db", side_effect=Exception("DB Error")), \
         patch("federated_pneumonia_detection.src.api.services.startup._init_ws") as mock_init_ws:
        
        with pytest.raises(Exception, match="DB Error"):
            await initialize_services()
            
        mock_init_ws.assert_not_called()

@pytest.mark.asyncio
async def test_shutdown_services(mock_app):
    """Test shutdown of all services."""
    with patch("federated_pneumonia_detection.src.api.services.startup._shut_db") as mock_shut_db, \
         patch("federated_pneumonia_detection.src.api.services.startup._shut_chat") as mock_shut_chat, \
         patch("federated_pneumonia_detection.src.api.services.startup._shut_mcp", new_callable=AsyncMock) as mock_shut_mcp, \
         patch("federated_pneumonia_detection.src.api.services.startup._shut_wandb") as mock_shut_wandb:
        
        # Simulate initialized MCP manager
        from federated_pneumonia_detection.src.api.services import startup
        startup._mcp_manager = MagicMock()
        
        await shutdown_services(mock_app)
        
        mock_shut_db.assert_called_once()
        mock_shut_chat.assert_called_once_with(mock_app)
        mock_shut_mcp.assert_called_once()
        mock_shut_wandb.assert_called_once()

@pytest.mark.asyncio
async def test_initialize_chat_services_internal(mock_app):
    """Test the internal initialize_chat_services function in the new module."""
    from federated_pneumonia_detection.src.api.services.internals.chat import initialize_chat_services
    
    with patch("federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.rag.QueryEngine") as mock_qe, \
         patch("federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine.ArxivAugmentedEngine") as mock_ae, \
         patch("federated_pneumonia_detection.src.api.services.internals.chat.AgentFactory") as mock_af:
        
        qe_inst = MagicMock()
        ae_inst = MagicMock()
        af_inst = MagicMock()
        
        mock_qe.return_value = qe_inst
        mock_ae.return_value = ae_inst
        mock_af.return_value = af_inst
        
        qe, af = await initialize_chat_services(mock_app)
        
        assert qe == qe_inst
        assert af == af_inst
        assert mock_app.state.query_engine == qe_inst
        assert mock_app.state.agent_factory == af_inst

@pytest.mark.asyncio
async def test_initialize_analytics_services_internal(mock_app):
    """Test the internal initialize_analytics_services function in the new module."""
    from federated_pneumonia_detection.src.api.services.internals.analytics import initialize_analytics_services
    
    with patch("federated_pneumonia_detection.src.api.services.internals.analytics.CacheProvider") as mock_cache, \
         patch("federated_pneumonia_detection.src.api.services.internals.analytics.MetricsService"), \
         patch("federated_pneumonia_detection.src.api.services.internals.analytics.SummaryService"), \
         patch("federated_pneumonia_detection.src.api.services.internals.analytics.RankingService"), \
         patch("federated_pneumonia_detection.src.api.services.internals.analytics.ExportService"), \
         patch("federated_pneumonia_detection.src.api.services.internals.analytics.BackfillService"), \
         patch("federated_pneumonia_detection.src.api.services.internals.analytics.AnalyticsFacade") as mock_facade:
        
        facade_inst = MagicMock()
        mock_facade.return_value = facade_inst
        
        facade = initialize_analytics_services(mock_app)
        
        assert facade == facade_inst
        assert mock_app.state.analytics == facade_inst
