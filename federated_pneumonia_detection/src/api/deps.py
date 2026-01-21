"""FastAPI dependency injection functions.

Provides singleton instances and database sessions for endpoint use.
"""

import logging
from typing import TYPE_CHECKING, Optional

from sqlalchemy.orm import Session

from federated_pneumonia_detection.config.config_manager import (
    ConfigManager,
    get_config_manager,
)
from federated_pneumonia_detection.src.boundary.CRUD.run import RunCRUD
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import RunMetricCRUD
from federated_pneumonia_detection.src.boundary.engine import get_session

if TYPE_CHECKING:
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.arxiv_mcp import (
        MCPManager,
    )
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.rag import (
        QueryEngine,
    )
    from federated_pneumonia_detection.src.control.model_inferance import (
        InferenceEngine,
        InferenceService,
    )

logger = logging.getLogger(__name__)


def get_db() -> Session:
    """Get a database session from the global connection pool.

    This is a FastAPI dependency that provides a database session for each request.
    The session is automatically closed after the request completes via the try/finally
    block, ensuring proper connection pool management.

    Yields:
        Session: SQLAlchemy database session from the connection pool
    """
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def get_config() -> ConfigManager:
    """Get the configuration manager"""
    return get_config_manager()


def get_experiment_crud() -> RunCRUD:
    """Get the experiment CRUD"""
    return RunCRUD()


def get_run_metric_crud() -> RunMetricCRUD:
    """Get the run metric CRUD"""
    return RunMetricCRUD()


_query_engine = None
_mcp_manager = None


def get_query_engine() -> Optional["QueryEngine"]:
    """Get or create QueryEngine singleton.

    Returns None if PostgreSQL database is unavailable (graceful degradation).
    """
    global _query_engine
    if _query_engine is None:
        try:
            from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.rag import (
                QueryEngine,
            )

            _query_engine = QueryEngine()
            logger.info("QueryEngine initialized successfully")
        except Exception as e:
            logger.warning(
                f"QueryEngine initialization failed (database unavailable): {e}",
            )
            return None
    return _query_engine


def get_mcp_manager() -> "MCPManager":
    """Get MCPManager singleton."""
    global _mcp_manager
    if _mcp_manager is None:
        from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.arxiv_mcp import (
            MCPManager,
        )

        _mcp_manager = MCPManager.get_instance()
        logger.info("MCPManager initialized successfully")
    return _mcp_manager


def get_inference_service() -> "InferenceService":
    """Get InferenceService singleton for X-ray inference.

    The service handles lazy loading of the model and clinical agent.
    """
    from federated_pneumonia_detection.src.control.model_inferance import (
        get_inference_service as _get_service,
    )

    return _get_service()


def get_inference_engine() -> Optional["InferenceEngine"]:
    """Get the InferenceEngine singleton.

    Returns None if the model cannot be loaded.
    """
    from federated_pneumonia_detection.src.control.model_inferance import (
        get_inference_engine as _get_engine,
    )

    return _get_engine()


def get_clinical_agent():
    """Get the
    singleton.

    Returns None if the agent cannot be initialized.
    """
    from federated_pneumonia_detection.src.control.model_inferance import (
        get_clinical_agent as _get_agent,
    )

    return _get_agent()
