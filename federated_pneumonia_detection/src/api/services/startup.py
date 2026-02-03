"""
Startup and shutdown service orchestration.

Manages initialization and cleanup of:
- Database tables
- MCP Manager (arxiv integration)
- W&B inference tracker
- WebSocket metrics server
"""

import logging

from federated_pneumonia_detection.src.api.services.internals.analytics import (
    initialize_analytics_services as _init_analytics,
)
from federated_pneumonia_detection.src.api.services.internals.chat import (
    initialize_chat_services as _init_chat,
)
from federated_pneumonia_detection.src.api.services.internals.chat import (
    shutdown_chat_services as _shut_chat,
)
from federated_pneumonia_detection.src.api.services.internals.database import (
    initialize_database as _init_db,
)
from federated_pneumonia_detection.src.api.services.internals.database import (
    shutdown_database as _shut_db,
)
from federated_pneumonia_detection.src.api.services.internals.mcp import (
    initialize_mcp_manager as _init_mcp,
)
from federated_pneumonia_detection.src.api.services.internals.mcp import (
    shutdown_mcp_manager as _shut_mcp,
)
from federated_pneumonia_detection.src.api.services.internals.wandb import (
    initialize_wandb_tracker as _init_wandb,
)
from federated_pneumonia_detection.src.api.services.internals.wandb import (
    shutdown_wandb_tracker as _shut_wandb,
)
from federated_pneumonia_detection.src.api.services.internals.websocket import (
    initialize_websocket_server as _init_ws,
)

logger = logging.getLogger(__name__)

# Module-level singleton references
_mcp_manager = None
_query_engine = None
_agent_factory = None


async def initialize_services(app=None) -> None:
    """
    Initialize all application services at startup.

    Order:
    1. Database tables (critical - raises on failure)
    2. WebSocket server (optional - logs warning on failure)
    3. MCP Manager (optional - logs warning on failure)
    4. Chat services (ArxivEngine, QueryEngine - optional, logs warning)
    5. Analytics services (optional - logs warning on failure)
    6. W&B tracker (optional - logs warning on failure)

    Args:
        app: FastAPI app instance for storing services in app.state
    """
    global _mcp_manager, _query_engine, _agent_factory

    # 1. Database (critical)
    _init_db()

    # 2. WebSocket server (optional, background thread)
    _init_ws()

    # 3. MCP Manager (optional)
    _mcp_manager = await _init_mcp()

    # 4. Chat services (optional - for performance optimization)
    _query_engine, _agent_factory = await _init_chat(app)

    # 5. Analytics services (optional - for performance optimization)
    _init_analytics(app)

    # 6. W&B tracker (optional)
    _init_wandb()


async def shutdown_services(app=None) -> None:
    """
    Clean up all application services at shutdown.

    Order:
    1. Database connections
    2. Chat services (HistoryManager connection pools)
    3. MCP Manager
    4. W&B tracker

    Args:
        app: FastAPI app instance for accessing app.state
    """
    global _mcp_manager, _query_engine, _agent_factory

    # 1. Database
    _shut_db()

    # 2. Chat services (cleanup connection pools)
    _shut_chat(app)

    # 3. MCP Manager
    if _mcp_manager is not None:
        await _shut_mcp(_mcp_manager)

    # 4. W&B tracker
    _shut_wandb()


# --- Internal helper wrappers for legacy compatibility (if needed) ---


def _initialize_database() -> None:
    """Legacy wrapper for database initialization."""
    _init_db()


async def _initialize_mcp_manager():
    """Legacy wrapper for MCP manager initialization."""
    return await _init_mcp()


def _initialize_wandb_tracker() -> None:
    """Legacy wrapper for W&B tracker initialization."""
    _init_wandb()


def _shutdown_database() -> None:
    """Legacy wrapper for database shutdown."""
    _shut_db()


async def _shutdown_mcp_manager(mcp_manager) -> None:
    """Legacy wrapper for MCP manager shutdown."""
    await _shut_mcp(mcp_manager)


def _shutdown_wandb_tracker() -> None:
    """Legacy wrapper for W&B tracker shutdown."""
    _shut_wandb()


async def _initialize_chat_services(app):
    """Legacy wrapper for chat services initialization."""
    return await _init_chat(app)


def _initialize_analytics_services(app):
    """Legacy wrapper for analytics services initialization."""
    return _init_analytics(app)


def _shutdown_chat_services(app) -> None:
    """Legacy wrapper for chat services shutdown."""
    _shut_chat(app)
