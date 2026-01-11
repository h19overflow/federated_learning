from federated_pneumonia_detection.src.boundary.engine import get_session
from sqlalchemy.orm import Session
from federated_pneumonia_detection.config.config_manager import (
    get_config_manager,
    ConfigManager,
)
from federated_pneumonia_detection.src.boundary.CRUD.run import RunCRUD
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import RunMetricCRUD
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_db() -> Session:
    """Get a database session"""
    session = get_session()
    try:
        return session
    except Exception as e:
        raise e


def get_config() -> ConfigManager:
    """Get the configuration manager"""
    return get_config_manager()


def get_experiment_crud() -> RunCRUD:
    """Get the experiment CRUD"""
    return RunCRUD()


def get_run_metric_crud() -> RunMetricCRUD:
    """Get the run metric CRUD"""
    return RunMetricCRUD()


# ==============================================================================
# CHAT ENGINE DEPENDENCIES
# ==============================================================================

_query_engine = None
_mcp_manager = None
_arxiv_engine: Optional["ArxivAugmentedEngine"] = None


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


# ==============================================================================
# INFERENCE DEPENDENCIES
# ==============================================================================

def get_inference_service() -> "InferenceService":
    """Get an InferenceService instance for X-ray inference.

    The service handles lazy loading of the model and clinical agent.
    """
    from federated_pneumonia_detection.src.boundary.inference_service import (
        get_inference_service as _get_service,
    )
    return _get_service()


def get_inference_engine() -> Optional["InferenceEngine"]:
    """Get the InferenceEngine singleton.

    Returns None if the model cannot be loaded.
    """
    from federated_pneumonia_detection.src.boundary.inference_service import (
        get_inference_engine as _get_engine,
    )
    return _get_engine()


def get_clinical_agent() -> Optional["ClinicalInterpretationAgent"]:
    """Get the ClinicalInterpretationAgent singleton.

    Returns None if the agent cannot be initialized.
    """
    from federated_pneumonia_detection.src.boundary.inference_service import (
        get_clinical_agent as _get_agent,
    )
    return _get_agent()


