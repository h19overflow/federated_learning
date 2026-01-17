from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.api.deps import get_config
from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
    ConfigurationUpdateRequest,
)

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

    try:
        current_config = config.to_dict()
        logger.info(
            f"Retrieved current config: epochs={current_config.get('experiment', {}).get('epochs')}, "
            f"batch_size={current_config.get('experiment', {}).get('batch_size')}, "
            f"learning_rate={current_config.get('experiment', {}).get('learning_rate')}"
        )

        return {
            "config": current_config,
            "config_path": str(config.config_path),
        }

    except Exception as e:
        logger.error(f"Failed to retrieve current configuration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to retrieve configuration", "message": str(e)}
        )


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

    try:
        config_data = configuration.model_dump(exclude_none=True)

        if not config_data:
            logger.warning("Empty configuration update request received")
            return {
                "message": "No configuration changes provided",
                "updated_fields": 0,
                "fields": [],
                "verified_values": {},
                "config_path": str(config.config_path),
            }

        flattened = config.flatten_config(config_data)

        updated_fields = []
        for key_path, value in flattened.items():
            try:
                config.set(key_path, value)
                updated_fields.append(key_path)
            except Exception as field_error:
                logger.warning(f"Failed to set field '{key_path}': {str(field_error)}")

        if not updated_fields:
            raise ValueError("Failed to update any configuration fields")

        config.save()
        config.reload()

        verification = {}
        for key in ["experiment.epochs", "experiment.batch_size", "experiment.learning_rate"]:
            if key in flattened:
                try:
                    verification[key.split('.')[1]] = config.get(key)
                except Exception as verify_error:
                    logger.warning(f"Failed to verify '{key}': {str(verify_error)}")

        logger.info(f"Configuration updated: {len(updated_fields)} field(s) modified")

        return {
            "message": "Configuration updated successfully",
            "updated_fields": len(updated_fields),
            "fields": updated_fields,
            "verified_values": verification,
            "config_path": str(config.config_path),
        }

    except ValueError as e:
        logger.warning(f"Configuration update validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "Configuration validation failed", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Configuration update error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to update configuration", "message": str(e)}
        )
