import logging

logger = logging.getLogger(__name__)

async def initialize_mcp_manager():
    """Initialize MCP manager for arxiv integration."""
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.arxiv_mcp import (  # noqa: E501
        MCPManager,
    )

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

async def shutdown_mcp_manager(mcp_manager) -> None:
    """Shutdown MCP manager."""
    try:
        await mcp_manager.shutdown()
        logger.info("MCP manager shutdown complete")
    except Exception as e:
        logger.warning(
            f"MCP manager shutdown had issues: {e} "
            "(this is non-fatal, app still shutting down)",
        )
