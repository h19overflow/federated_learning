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
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine import (
    ArxivAugmentedEngine,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.arxiv_mcp import (
    MCPManager,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.rag import (
    QueryEngine,
)
from federated_pneumonia_detection.src.control.dl_model.internals.data.wandb_inference_tracker import (
    get_wandb_tracker,
)

logger = logging.getLogger(__name__)

# Module-level singleton references
_mcp_manager: MCPManager | None = None
_arxiv_engine: ArxivAugmentedEngine | None = None
_query_engine: QueryEngine | None = None


async def initialize_services(app=None) -> None:
    """
    Initialize all application services at startup.

    Order:
    1. Database tables (critical - raises on failure)
    2. WebSocket server (optional - logs warning on failure)
    3. MCP Manager (optional - logs warning on failure)
    4. Chat services (ArxivEngine, QueryEngine - optional, logs warning)
    5. W&B tracker (optional - logs warning on failure)

    Args:
        app: FastAPI app instance for storing services in app.state
    """
    global _mcp_manager, _arxiv_engine, _query_engine

    # 1. Database (critical)
    _initialize_database()

    # 2. WebSocket server (optional, background thread)
    start_websocket_server_thread()

    # 3. MCP Manager (optional)
    _mcp_manager = await _initialize_mcp_manager()

    # 4. Chat services (optional - for performance optimization)
    _arxiv_engine, _query_engine = await _initialize_chat_services(app)

    # 5. W&B tracker (optional)
    _initialize_wandb_tracker()


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
    global _mcp_manager, _arxiv_engine, _query_engine

    # 1. Database
    _shutdown_database()

    # 2. Chat services (cleanup connection pools)
    _shutdown_chat_services(app)

    # 3. MCP Manager
    if _mcp_manager is not None:
        await _shutdown_mcp_manager(_mcp_manager)

    # 4. W&B tracker
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
            "MCP manager initialized successfully (arxiv integration available)",
        )
    except ConnectionError as e:
        logger.warning(
            f"MCP initialization failed - network issue: {e} "
            "(arxiv search will be unavailable)",
        )
    except ImportError as e:
        logger.warning(
            f"MCP initialization failed - missing dependency: {e} "
            "(arxiv search disabled)",
        )
    except Exception as e:
        logger.warning(
            f"MCP initialization failed (unexpected error): {e} "
            "(arxiv search will be unavailable, but app continues)",
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
                "W&B inference tracker initialized (experiment tracking enabled)",
            )
        else:
            logger.warning(
                "W&B tracker rejected configuration (check credentials). "
                "Experiment tracking will be unavailable.",
            )
    except ConnectionError as e:
        logger.warning(
            f"W&B connection failed: {e} "
            "(experiment tracking disabled, but training continues)",
        )
    except ImportError as e:
        logger.warning(
            f"W&B not installed: {e} (install with: pip install wandb) "
            "(experiment tracking disabled)",
        )
    except Exception as e:
        logger.warning(
            f"W&B initialization failed (unexpected): {e} "
            "(experiment tracking will be unavailable)",
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
            "(this is non-fatal, app still shutting down)",
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
            "(this is non-fatal, app still shutting down)",
        )


async def _initialize_chat_services(
    app,
) -> tuple[
    ArxivAugmentedEngine | None,
    QueryEngine | None,
]:
    """
    Initialize chat services (ArxivEngine and QueryEngine) at startup.

    These services are heavy to initialize (LLMs, vector stores, BM25),
    so initializing them once at startup significantly reduces first-message latency.

    Both services are optional - if initialization fails, endpoints will fall back
    to lazy initialization or graceful degradation.

    Args:
        app: FastAPI app instance for storing services in app.state

    Returns:
        Tuple of (arxiv_engine, query_engine) - either initialized instances or None
    """
    arxiv_engine = None
    query_engine = None

    # Initialize QueryEngine first (needed by ArxivEngine)
    try:
        logger.info("[Startup] Initializing QueryEngine (RAG system)...")
        query_engine = QueryEngine(max_history=10)
        logger.info("[Startup] QueryEngine initialized successfully")
    except Exception as e:
        logger.warning(
            f"[Startup] QueryEngine initialization failed: {e} "
            "(RAG features unavailable, will fall back to lazy init)",
        )

    # Initialize ArxivEngine (main research engine)
    try:
        logger.info("[Startup] Initializing ArxivEngine (research engine)...")
        arxiv_engine = ArxivAugmentedEngine(max_history=10)
        logger.info("[Startup] ArxivEngine initialized successfully")
    except Exception as e:
        logger.warning(
            f"[Startup] ArxivEngine initialization failed: {e} "
            "(arxiv features unavailable, will fall back to lazy init)",
        )

    # Store in app.state if available
    if app is not None:
        app.state.arxiv_engine = arxiv_engine
        app.state.query_engine = query_engine
        logger.info("[Startup] Chat services stored in app.state")

    return arxiv_engine, query_engine


def _shutdown_chat_services(app) -> None:
    """
    Shutdown chat services and cleanup connection pools.

    The HistoryManager uses connection pooling which needs proper cleanup.

    Args:
        app: FastAPI app instance for accessing app.state
    """
    if app is None:
        return

    # Cleanup ArxivEngine (includes HistoryManager)
    if hasattr(app.state, "arxiv_engine") and app.state.arxiv_engine is not None:
        try:
            # Access the history manager and close its connection pool
            if hasattr(app.state.arxiv_engine, "_history_manager"):
                app.state.arxiv_engine._history_manager.close()
                logger.info(
                    "[Shutdown] ArxivEngine HistoryManager connection pool closed"
                )
        except Exception as e:
            logger.warning(f"[Shutdown] Error closing ArxivEngine: {e}")

    # Cleanup QueryEngine (includes HistoryManager)
    if hasattr(app.state, "query_engine") and app.state.query_engine is not None:
        try:
            if hasattr(app.state.query_engine, "history_manager"):
                app.state.query_engine.history_manager.close()
                logger.info(
                    "[Shutdown] QueryEngine HistoryManager connection pool closed"
                )
        except Exception as e:
            logger.warning(f"[Shutdown] Error closing QueryEngine: {e}")

    logger.info("[Shutdown] Chat services shutdown complete")
