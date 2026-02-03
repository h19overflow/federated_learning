import logging

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.factory import (  # noqa: E501
    AgentFactory,
)

logger = logging.getLogger(__name__)

async def initialize_chat_services(app):
    """
    Initialize chat services (ArxivEngine, QueryEngine, and AgentFactory) at startup.

    These services are heavy to initialize (LLMs, vector stores, BM25),
    so initializing them once at startup significantly reduces first-message latency.

    All services are optional - if initialization fails, endpoints will fall back
    to lazy initialization or graceful degradation.

    Args:
        app: FastAPI app instance for storing services in app.state

    Returns:
        Tuple of (query_engine, agent_factory) - either initialized instances or None
    """
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_engine import (  # noqa: E501
        ArxivAugmentedEngine,
    )
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.rag import (  # noqa: E501
        QueryEngine,
    )

    arxiv_engine = None
    query_engine = None
    agent_factory = None

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
        arxiv_engine = ArxivAugmentedEngine(max_history=10, query_engine=query_engine)
        logger.info("[Startup] ArxivEngine initialized successfully")
    except Exception as e:
        logger.warning(
            f"[Startup] ArxivEngine initialization failed: {e} "
            "(arxiv features unavailable, will fall back to lazy init)",
        )

    # Initialize AgentFactory (uses pre-initialized arxiv_engine for performance)
    try:
        logger.info("[Startup] Initializing AgentFactory...")

        # Create a temporary app.state object to pass arxiv_engine
        class TempState:
            arxiv_engine: ArxivAugmentedEngine | None = None

        temp_state = TempState()
        temp_state.arxiv_engine = arxiv_engine

        agent_factory = AgentFactory(app_state=temp_state)
        logger.info("[Startup] AgentFactory initialized successfully")
    except Exception as e:
        logger.warning(
            f"[Startup] AgentFactory initialization failed: {e} "
            "(chat endpoints will create factory on-demand)",
        )

    # Store in app.state if available
    if app is not None:
        app.state.query_engine = query_engine
        app.state.agent_factory = agent_factory
        logger.info("[Startup] Chat services stored in app.state")

    return query_engine, agent_factory

def shutdown_chat_services(app) -> None:
    """
    Shutdown chat services and cleanup connection pools.

    The HistoryManager uses connection pooling which needs proper cleanup.

    Args:
        app: FastAPI app instance for accessing app.state
    """
    if app is None:
        return

    # Cleanup ArxivEngine (via agent_factory)
    if hasattr(app.state, "agent_factory") and app.state.agent_factory is not None:
        try:
            # Access the engine through the factory's cached agent
            if (
                hasattr(app.state.agent_factory, "_research_agent")
                and app.state.agent_factory._research_agent is not None
            ):
                engine = app.state.agent_factory._research_agent
                if hasattr(engine, "_history_manager"):
                    engine._history_manager.close()
                    logger.info(
                        "[Shutdown] ArxivEngine HistoryManager connection pool closed"
                    )
        except Exception as e:
            logger.warning(f"[Shutdown] Error closing ArxivEngine via factory: {e}")

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
