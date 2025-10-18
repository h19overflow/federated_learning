from federated_pneumonia_detection.src.boundary.engine import get_session
from sqlalchemy.orm import Session
from federated_pneumonia_detection.config.config_manager import get_config_manager, ConfigManager
from federated_pneumonia_detection.src.boundary.CRUD.run import RunCRUD
from federated_pneumonia_detection.src.boundary.CRUD.run_configuration import RunConfigurationCRUD
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import RunMetricCRUD
from federated_pneumonia_detection.src.boundary.CRUD.run_artifact import RunArtifactCRUD

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

def get_run_configuration_crud() -> RunConfigurationCRUD:
    """Get the run configuration CRUD"""
    return RunConfigurationCRUD()

def get_run_metric_crud() -> RunMetricCRUD:
    """Get the run metric CRUD"""
    return RunMetricCRUD()

def get_run_artifact_crud() -> RunArtifactCRUD:
    """Get the run artifact CRUD"""
    return RunArtifactCRUD()
