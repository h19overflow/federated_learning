from federated_pneumonia_detection.src.api.deps import get_config
from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.api.endpoints.configuration_settings.schemas import (
    ConfigurationUpdateRequest,
)
from fastapi import APIRouter, Depends
from typing import Dict, Any

router = APIRouter(
    prefix="/config",
    tags=["config"],
)


@router.get("/current")
async def get_current_settings(
    config: ConfigManager = Depends(get_config),
) -> Dict[str, Any]:
    """
    Get current configuration values.

    Returns:
        Dict with current configuration
    """
    from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
    logger = get_logger(__name__)

    current_config = config.to_dict()
    logger.info(f"Retrieved current config: epochs={current_config.get('experiment', {}).get('epochs')}, "
                f"batch_size={current_config.get('experiment', {}).get('batch_size')}, "
                f"learning_rate={current_config.get('experiment', {}).get('learning_rate')}")

    return {
        "config": current_config,
        "config_path": str(config.config_path),
    }


@router.post("/update")
async def update_settings(
    configuration: ConfigurationUpdateRequest,
    config: ConfigManager = Depends(get_config),
) -> Dict[str, Any]:
    """
    Update configuration values.

    Accepts partial configuration updates. Only provided fields will be updated,
    allowing overrides of specific settings while keeping others as presets.

    Args:
        configuration: ConfigurationUpdateRequest with optional nested config sections
        config: ConfigManager instance

    Returns:
        Dict with success message and updated settings count
    """
    from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
    logger = get_logger(__name__)

    config_data = configuration.model_dump(exclude_none=True)
    flattened = config._flatten_config(config_data)

    logger.info(f"Updating {len(flattened)} configuration fields: {list(flattened.keys())}")

    # Log key values being updated
    if 'experiment.epochs' in flattened:
        logger.info(f"Setting experiment.epochs = {flattened['experiment.epochs']}")
    if 'experiment.batch_size' in flattened:
        logger.info(f"Setting experiment.batch_size = {flattened['experiment.batch_size']}")
    if 'experiment.learning_rate' in flattened:
        logger.info(f"Setting experiment.learning_rate = {flattened['experiment.learning_rate']}")

    updated_count = 0
    for key_path, value in flattened.items():
        config.set(key_path, value)
        updated_count += 1

    config.save()
    from pathlib import Path
    abs_path = Path(config.config_path).resolve()
    logger.info(f"Configuration saved to {config.config_path} (absolute: {abs_path})")

    # Verify save by reloading
    config.reload()
    verification = {}
    if 'experiment.epochs' in flattened:
        verification['epochs'] = config.get('experiment.epochs')
    if 'experiment.batch_size' in flattened:
        verification['batch_size'] = config.get('experiment.batch_size')
    if 'experiment.learning_rate' in flattened:
        verification['learning_rate'] = config.get('experiment.learning_rate')

    logger.info(f"Verified saved values: {verification}")

    return {
        "message": "Configuration updated successfully",
        "updated_fields": updated_count,
        "fields": list(flattened.keys()),
        "verified_values": verification,
        "config_path": str(config.config_path),
    }
