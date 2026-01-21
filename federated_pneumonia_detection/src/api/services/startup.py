"""
Startup and shutdown service orchestration.

Manages initialization and cleanup of:
- Database tables
- MCP Manager (arxiv integration)
- W&B inference tracker
- WebSocket metrics server
"""

import logging

from federated_pneumonia_detection.src.api.endpoints.streaming.websocket_server import (
    start_websocket_server_thread,
)
from federated_pneumonia_detection.src.boundary.engine import create_tables
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.arxiv_mcp import (
    MCPManager,
)
from federated_pneumonia_detection.src.control.dl_model.internals.data.wandb_inference_tracker import (
    get_wandb_tracker,
)

logger = logging.getLogger(__name__)

# Module-level singleton reference
_mcp_manager: MCPManager | None = None


async def initialize_services() -> None:
    """
    Initialize all application services at startup.

    Order:
    1. Database tables (critical - raises on failure)
    2. WebSocket server (optional - logs warning on failure)
    3. MCP Manager (optional - logs warning on failure)
    4. W&B tracker (optional - logs warning on failure)
    """
    global _mcp_manager

    # 1. Database (critical)
    _initialize_database()

    # 2. WebSocket server (optional, background thread)
    start_websocket_server_thread()

    # 3. MCP Manager (optional)
    _mcp_manager = await _initialize_mcp_manager()

    # 4. W&B tracker (optional)
    _initialize_wandb_tracker()


async def shutdown_services() -> None:
    """
    Clean up all application services at shutdown.

    Order:
    1. Database connections
    2. MCP Manager
    3. W&B tracker
    """
    global _mcp_manager

    # 1. Database
    _shutdown_database()

    # 2. MCP Manager
    if _mcp_manager is not None:
        await _shutdown_mcp_manager(_mcp_manager)

    # 3. W&B tracker
    _shutdown_wandb_tracker()


def _initialize_database() -> None:
    """Create database tables. Raises on failure (critical service)."""
    try:
        logger.info("Ensuring database tables exist...")
        create_tables()
        logger.info("Database tables verified/created")
    except Exception as e:
        logger.critical(f"DATABASE INITIALIZATION FAILED: {e}")
        logger.critical("Cannot proceed with startup. Shutting down.")
        raise


async def _initialize_mcp_manager() -> MCPManager:
    """Initialize MCP manager for arxiv integration."""
    mcp_manager = MCPManager.get_instance()

    try:
        await mcp_manager.initialize()
        logger.info(
            "MCP manager initialized successfully (arxiv integration available)"
        )
    except ConnectionError as e:
        logger.warning(
            f"MCP initialization failed - network issue: {e} "
            "(arxiv search will be unavailable)"
        )
    except ImportError as e:
        logger.warning(
            f"MCP initialization failed - missing dependency: {e} "
            "(arxiv search disabled)"
        )
    except Exception as e:
        logger.warning(
            f"MCP initialization failed (unexpected error): {e} "
            "(arxiv search will be unavailable, but app continues)"
        )

    return mcp_manager


def _initialize_wandb_tracker() -> None:
    """Initialize W&B inference tracker for experiment tracking."""
    try:
        tracker = get_wandb_tracker()
        if tracker.initialize(
            entity="projectontheside25-multimedia-university",
            project="FYP2",
            job_type="inference",
        ):
            logger.info(
                "W&B inference tracker initialized (experiment tracking enabled)"
            )
        else:
            logger.warning(
                "W&B tracker rejected configuration (check credentials). "
                "Experiment tracking will be unavailable."
            )
    except ConnectionError as e:
        logger.warning(
            f"W&B connection failed: {e} "
            "(experiment tracking disabled, but training continues)"
        )
    except ImportError as e:
        logger.warning(
            f"W&B not installed: {e} (install with: pip install wandb) "
            "(experiment tracking disabled)"
        )
    except Exception as e:
        logger.warning(
            f"W&B initialization failed (unexpected): {e} "
            "(experiment tracking will be unavailable)"
        )


def _shutdown_database() -> None:
    """Dispose database connections."""
    try:
        from federated_pneumonia_detection.src.boundary.engine import dispose_engine

        dispose_engine()
        logger.info("Database connections disposed")
    except Exception as e:
        logger.warning(f"Error disposing database connections: {e}")


async def _shutdown_mcp_manager(mcp_manager: MCPManager) -> None:
    """Shutdown MCP manager."""
    try:
        await mcp_manager.shutdown()
        logger.info("MCP manager shutdown complete")
    except Exception as e:
        logger.warning(
            f"MCP manager shutdown had issues: {e} "
            "(this is non-fatal, app still shutting down)"
        )


def _shutdown_wandb_tracker() -> None:
    """Finish W&B tracking session."""
    try:
        tracker = get_wandb_tracker()
        tracker.finish()
        logger.info("W&B inference tracker shutdown complete")
    except Exception as e:
        logger.warning(
            f"W&B tracker shutdown had issues: {e} "
            "(this is non-fatal, app still shutting down)"
        )
