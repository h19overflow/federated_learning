import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys

# Mock the missing module before importing MCPManager
mock_mcp_client = MagicMock()
sys.modules["langchain_mcp_adapters"] = MagicMock()
sys.modules["langchain_mcp_adapters.client"] = mock_mcp_client

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.arxiv_mcp import (
    MCPManager,
)


@pytest.mark.asyncio
async def test_mcp_manager_singleton():
    """Test that MCPManager is a singleton."""
    manager1 = MCPManager.get_instance()
    manager2 = MCPManager.get_instance()
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_mcp_manager_initialization_success():
    """Test successful initialization of MCPManager."""
    # Reset singleton for clean test
    MCPManager._instance = None
    manager = MCPManager.get_instance()

    mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]
    mock_client_instance = AsyncMock()
    mock_client_instance.get_tools.return_value = mock_tools

    with patch.object(
        mock_mcp_client, "MultiServerMCPClient", return_value=mock_client_instance
    ):
        await manager.initialize()

        assert manager._client is not None
        assert manager.is_available is True
        assert manager.get_arxiv_tools() == mock_tools


@pytest.mark.asyncio
async def test_mcp_manager_initialization_failure():
    """Test initialization failure of MCPManager."""
    MCPManager._instance = None
    manager = MCPManager.get_instance()

    with patch.object(
        mock_mcp_client,
        "MultiServerMCPClient",
        side_effect=Exception("Connection failed"),
    ):
        await manager.initialize()
        assert manager.is_available is False
        assert manager.get_arxiv_tools() == []


@pytest.mark.asyncio
async def test_mcp_manager_shutdown():
    """Test shutdown of MCPManager."""
    MCPManager._instance = None
    manager = MCPManager.get_instance()
    manager._is_available = True
    manager._client = MagicMock()
    manager._tools = ["tool1"]

    await manager.shutdown()
    assert manager.is_available is False
    assert manager._client is None
    assert manager._tools == []


@pytest.mark.asyncio
async def test_mcp_manager_context_manager():
    """Test MCPManager as an async context manager."""
    MCPManager._instance = None
    manager = MCPManager.get_instance()

    mock_client_instance = AsyncMock()
    mock_client_instance.get_tools.return_value = ["tool1"]

    with patch.object(
        mock_mcp_client, "MultiServerMCPClient", return_value=mock_client_instance
    ):
        async with manager as m:
            assert m is manager
            assert m.is_available is True

        assert manager.is_available is False
