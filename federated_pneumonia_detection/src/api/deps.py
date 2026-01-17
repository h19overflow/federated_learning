from federated_pneumonia_detection.src.boundary.engine import get_session
from sqlalchemy.orm import Session
from federated_pneumonia_detection.config.config_manager import (
    get_config_manager,
    ConfigManager,
)
from federated_pneumonia_detection.src.boundary.CRUD.run import RunCRUD
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import RunMetricCRUD
from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.retriver import QueryEngine
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.mcp_manager import MCPManager
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent import ArxivAugmentedEngine
    from federated_pneumonia_detection.src.control.model_inferance.inference_engine import InferenceEngine
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.clinical import ClinicalInterpretationAgent

logger = logging.getLogger(__name__)


def get_db() -> Session:
    """Get a database session from the global connection pool.

    This is a FastAPI dependency that provides a database session for each request.
    The session is automatically closed after the request completes via the try/finally
    block, ensuring proper connection pool management.

    The session is created from the global pooled engine (see engine.py's get_session()),
    which uses SQLAlchemy's QueuePool for efficient connection reuse across requests.

    Usage Example:
        ```python
        from fastapi import Depends
        from sqlalchemy.orm import Session

        @router.get("/api/runs/{run_id}")
        async def get_run(
            run_id: int,
            db: Session = Depends(get_db)  # Automatically closed after request
        ):
            return crud.get_run(db, run_id)
        ```

    Yields:
        Session: SQLAlchemy database session from the connection pool

    Note:
        FastAPI calls this generator function as a dependency and will execute the
        finally block after the route handler completes, ensuring session.close() is
        always called even if an exception occurs during request processing.
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
_arxiv_engine: Optional["ArxivAugmentedEngine"] = None
_inference_engine: Optional["InferenceEngine"] = None
_clinical_agent: Optional["ClinicalInterpretationAgent"] = None


def get_query_engine() -> Optional["QueryEngine"]:
    """
    Get or create QueryEngine singleton.

    Returns None if PostgreSQL database is unavailable (graceful degradation).
    """
    global _query_engine
    if _query_engine is None:
        try:
            from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.retriver import (
                QueryEngine,
            )
            _query_engine = QueryEngine()
            logger.info("QueryEngine initialized successfully")
        except Exception as e:
            logger.warning(f"QueryEngine initialization failed (database unavailable): {e}")
            return None
    return _query_engine


def get_mcp_manager() -> "MCPManager":
    """Get MCPManager singleton."""
    global _mcp_manager
    if _mcp_manager is None:
        from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.mcp_manager import (
            MCPManager,
        )
        _mcp_manager = MCPManager.get_instance()
        logger.info("MCPManager initialized successfully")
    return _mcp_manager


def get_arxiv_engine() -> "ArxivAugmentedEngine":
    """Get or create ArxivAugmentedEngine singleton."""
    global _arxiv_engine
    if _arxiv_engine is None:
        from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent import (
            ArxivAugmentedEngine,
        )
        _arxiv_engine = ArxivAugmentedEngine()
        logger.info("ArxivAugmentedEngine initialized successfully")
    return _arxiv_engine

def get_inference_engine() -> Optional["InferenceEngine"]:
    """Get or create InferenceEngine singleton.

    Returns None if initialization fails (graceful degradation).
    """
    global _inference_engine

    if _inference_engine is None:
        try:
            from federated_pneumonia_detection.src.control.model_inferance.inference_engine import (
                InferenceEngine,
            )
            _inference_engine = InferenceEngine()
            logger.info("InferenceEngine initialized successfully")
        except Exception as e:
            logger.error(f"InferenceEngine initialization failed: {e}", exc_info=True)
            return None

    return _inference_engine


def get_clinical_agent() -> Optional["ClinicalInterpretationAgent"]:
    """Get or create ClinicalInterpretationAgent singleton.

    Returns None if initialization fails (graceful degradation).
    """
    global _clinical_agent

    if _clinical_agent is None:
        try:
            from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.clinical import (
                ClinicalInterpretationAgent,
            )
            _clinical_agent = ClinicalInterpretationAgent()
            logger.info("ClinicalInterpretationAgent initialized successfully")
        except Exception as e:
            logger.warning(f"ClinicalInterpretationAgent unavailable: {e}")
            return None

    return _clinical_agent


def is_model_loaded() -> bool:
    """Check if the inference model is loaded."""
    return _inference_engine is not None


def get_model_info() -> dict:
    """Get information about the loaded model."""
    if _inference_engine is None:
        return {
            "loaded": False,
            "version": None,
            "device": None,
        }
    return {
        "loaded": True,
        **_inference_engine.get_info(),
    }


