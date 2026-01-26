from typing import Any, Dict

from fastapi import APIRouter, Depends

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
        Dict with current configuration and path
    """
    return {
        "config": config.to_dict(),
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
        Dict with success message and updated settings metadata
    """
    config_data = configuration.model_dump(exclude_none=True)
    update_result = config.update_from_dict(config_data)

    return {
        "message": "Configuration updated successfully",
        "updated_fields": update_result["updated_count"],
        "fields": update_result["fields"],
        "verified_values": update_result["verified_values"],
        "config_path": str(config.config_path),
    }
