from federated_pneumonia_detection.src.api.deps import get_config
from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.api.endpoints.configruation_settings.schemas import (
    ConfigurationUpdateRequest,
)
from fastapi import APIRouter, Depends
from typing import Dict, Any

router = APIRouter(
    prefix="/configuration",
    tags=["configuration"],
)


@router.post("/set_configuration")
async def set_configuration(
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
    config_data = configuration.model_dump(exclude_none=True)
    flattened = config._flatten_config(config_data)

    updated_count = 0
    for key_path, value in flattened.items():
        config.set(key_path, value)
        updated_count += 1

    config.save()

    return {
        "message": "Configuration updated successfully",
        "updated_fields": updated_count,
        "fields": list(flattened.keys()),
    }
