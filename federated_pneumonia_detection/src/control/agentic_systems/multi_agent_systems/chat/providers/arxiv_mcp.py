"""
MCP Manager - Singleton for arxiv-mcp-server lifecycle management.

Manages the arxiv MCP server subprocess and exposes arxiv tools as LangChain tools.

Dependencies:
    - langchain_mcp_adapters.client.MultiServerMCPClient
    - arxiv-mcp-server CLI tool
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class MCPManager:
    """
    Singleton manager for MCP server connections.

    Handles arxiv-mcp-server lifecycle and tool exposure to LangChain agents.
    """

    _instance: Optional[MCPManager] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> MCPManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._client = None
        self._tools: List[BaseTool] = []
        self._is_available = False
        self._initialized = True

    @classmethod
    def get_instance(cls) -> MCPManager:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_available(self) -> bool:
        """Check if MCP server is connected and tools are available."""
        return self._is_available and len(self._tools) > 0

    async def initialize(self) -> None:
        """
        Start MCP server and initialize client connection.

        Connects to arxiv-mcp-server via stdio transport and loads available tools.
        """
        async with self._lock:
            if self._is_available:
                logger.info("MCP Manager already initialized")
                return

            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient

                server_config = {
                    "arxiv": {
                        "command": "arxiv-mcp-server",
                        "args": [],
                        "transport": "stdio",
                    },
                }

                self._client = MultiServerMCPClient(server_config)
                self._tools = await self._client.get_tools()
                self._is_available = True
            except Exception as e:
                logger.error(f"Failed to initialize MCP Manager: {e}")
                self._is_available = False

    async def shutdown(self) -> None:
        """
        Cleanly stop MCP server and close client connection.
        """
        async with self._lock:
            if not self._is_available:
                return

            try:
                self._client = None
                self._tools = []
                self._is_available = False
                logger.info("MCP Manager shutdown complete")

            except Exception as e:
                logger.error(f"Error during MCP Manager shutdown: {e}")
                self._is_available = False

    def get_arxiv_tools(self) -> List[BaseTool]:
        """
        Get arxiv tools as LangChain BaseTool instances.

        Returns:
            List of LangChain tools for arxiv operations.
            Empty list if server is unavailable.
        """
        if not self._is_available:
            logger.warning("MCP Manager not available, returning empty tools list")
            return []

        return self._tools

    async def __aenter__(self) -> MCPManager:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
