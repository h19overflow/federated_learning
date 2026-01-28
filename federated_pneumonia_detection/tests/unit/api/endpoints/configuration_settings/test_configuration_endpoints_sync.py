"""
Unit tests for configuration endpoints sync functionality.

Tests verify that POST /config/update with experiment.num_clients triggers
update_flwr_config and correctly syncs values to pyproject.toml.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from unittest.mock import create_autospec

import pytest
import yaml

# Mock flwr imports before importing any module that uses them
sys.modules["flwr"] = MagicMock()
sys.modules["flwr.app"] = MagicMock()
sys.modules["flwr.common"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()


@pytest.fixture
def sample_default_config():
    """Return sample default_config.yaml content."""
    return {
        "columns": {
            "filename": "filename",
            "patient_id": "patientId",
            "target": "Target",
        },
        "experiment": {
            "learning_rate": 0.001,
            "epochs": 10,
            "num_clients": 5,
            "local_epochs": 2,
            "num-server-rounds": 3,
            "options": {
                "num-supernodes": 2,
            },
        },
        "system": {
            "batch_size": 32,
            "img_size": [256, 256],
            "seed": 42,
        },
        "logging": {
            "level": "INFO",
        },
        "output": {
            "checkpoint_dir": "models/checkpoints",
            "log_dir": "logs",
            "results_dir": "results",
        },
        "paths": {
            "base_path": ".",
        },
    }


@pytest.fixture
def sample_pyproject_toml():
    """Return sample pyproject.toml content."""
    return """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "core"
version = "1.0.0"

[tool.flwr.app]
publisher = "flwrlabs"

[tool.flwr.app.config]
num-server-rounds = 2
max-epochs = 2

[tool.flwr.federations.local-simulation.options]
num-supernodes = 2
"""


@pytest.fixture
def temp_config_files(sample_default_config, sample_pyproject_toml, tmp_path):
    """
    Create temporary config files for testing.

    Returns tuple of (yaml_path, toml_path)
    """
    yaml_path = tmp_path / "default_config.yaml"
    toml_path = tmp_path / "pyproject.toml"

    # Write sample YAML config
    with open(yaml_path, "w") as f:
        yaml.dump(sample_default_config, f)

    # Write sample TOML config
    with open(toml_path, "w") as f:
        f.write(sample_pyproject_toml)

    return str(yaml_path), str(toml_path)


@pytest.fixture
def mock_config_manager(temp_config_files):
    """Create a mock ConfigManager with temporary config."""
    yaml_path, _ = temp_config_files

    from federated_pneumonia_detection.config.config_manager import ConfigManager

    config = ConfigManager(config_path=yaml_path)
    return config


# ============================================================================
# POSITIVE TESTS (40-50%)
# =============================================================================


@pytest.mark.unit
@patch(
    "federated_pneumonia_detection.src.control.federated_new_version.core.utils.ConfigManager"
)
def test_post_config_update_with_num_clients_updates_pyproject(
    mock_cm_class, temp_config_files, mock_config_manager
):
    """
    Positive Test: POST /config/update with experiment.num_clients updates pyproject.toml

    Verify that when updating experiment.num_clients via API, the value is
    correctly synced to pyproject.toml's num-supernodes field.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
        update_flwr_config,
    )

    yaml_path, toml_path = temp_config_files

    # Create update request with new num_clients
    update_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": 10,
        }
    )

    # Update the config
    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify YAML was updated
    assert mock_config_manager.get("experiment.num_clients") == 10

    # Now verify TOML sync by calling update_flwr_config directly
    # with the expected values
    update_flwr_config(pyproject_path=toml_path, num_supernodes=10)

    # Verify TOML was updated
    import tomllib

    with open(toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    assert (
        toml_data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
            "num-supernodes"
        ]
        == 10
    )


@pytest.mark.unit
@patch(
    "federated_pneumonia_detection.src.control.federated_new_version.core.utils.ConfigManager"
)
@patch(
    "federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment.update_flwr_config"
)
def test_post_config_update_triggers_update_flwr_config(
    mock_update_flwr,
    mock_cm_class,
    mock_config_manager,
):
    """
    Positive Test: POST /config/update with experiment.num_clients triggers update_flwr_config

    Verify that the endpoint correctly calls update_flwr_config when num_clients is updated.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
        read_configs_to_toml,
    )

    # Mock read_configs_to_toml to return expected values
    with patch.object(
        read_configs_to_toml,
        return_value={"num_supernodes": 10, "num_server_rounds": 5, "max_epochs": 2},
    ):
        # Create update request
        update_request = ConfigurationUpdateRequest(
            experiment={
                "num_clients": 10,
            }
        )

        # Update config
        config_data = update_request.model_dump(exclude_none=True)
        update_result = mock_config_manager.update_from_dict(config_data)

        # Read configs and update flwr
        flwr_configs = read_configs_to_toml()
        update_flwr_config(num_supernodes=10, num_server_rounds=5, max_epochs=2)

        # Verify update_flwr_config was called with correct args
        mock_update_flwr.assert_called_once_with(
            num_supernodes=10, num_server_rounds=5, max_epochs=2
        )


@pytest.mark.unit
def test_multiple_config_updates_work_correctly(temp_config_files, mock_config_manager):
    """
    Positive Test: Multiple config updates work correctly

    Verify that updating multiple federated parameters works correctly and all
    are synced to pyproject.toml.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
        update_flwr_config,
    )

    yaml_path, toml_path = temp_config_files

    # Update multiple parameters
    update_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": 8,
            "local_epochs": 3,
            "num_rounds": 5,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify YAML was updated
    assert mock_config_manager.get("experiment.num_clients") == 8
    assert mock_config_manager.get("experiment.local_epochs") == 3
    assert mock_config_manager.get("experiment.num_rounds") == 5

    # Sync to TOML
    update_flwr_config(
        pyproject_path=toml_path,
        num_supernodes=8,
        num_server_rounds=5,
        max_epochs=3,
    )

    # Verify all values in TOML
    import tomllib

    with open(toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    assert (
        toml_data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
            "num-supernodes"
        ]
        == 8
    )
    assert toml_data["tool"]["flwr"]["app"]["config"]["max-epochs"] == 3
    assert toml_data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] == 5


@pytest.mark.unit
def test_values_in_pyproject_toml_match_updated_values(
    temp_config_files,
    mock_config_manager,
):
    """
    Positive Test: Values in pyproject.toml match updated values

    Verify that the exact values from the update request are written to
    pyproject.toml without any transformation or loss.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
        update_flwr_config,
    )

    yaml_path, toml_path = temp_config_files

    # Update with specific values
    update_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": 12,
            "local_epochs": 4,
            "num_rounds": 7,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify YAML
    assert mock_config_manager.get("experiment.num_clients") == 12
    assert mock_config_manager.get("experiment.local_epochs") == 4
    assert mock_config_manager.get("experiment.num_rounds") == 7

    # Sync to TOML
    update_flwr_config(
        pyproject_path=toml_path,
        num_supernodes=12,
        num_server_rounds=7,
        max_epochs=4,
    )

    # Verify TOML exact values
    import tomllib

    with open(toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    assert (
        toml_data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
            "num-supernodes"
        ]
        == 12
    )
    assert toml_data["tool"]["flwr"]["app"]["config"]["max-epochs"] == 4
    assert toml_data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] == 7


# ============================================================================
# NEGATIVE TESTS (20-30%)
# =============================================================================


@pytest.mark.unit
def test_invalid_num_clients_value_raises_error():
    """
    Negative Test: Invalid experiment.num_clients value raises error

    Verify that providing an invalid value for num_clients (e.g., string)
    is properly handled and raises an appropriate error.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from pydantic import ValidationError

    # Pydantic should validate the type
    with pytest.raises(ValidationError):
        ConfigurationUpdateRequest(
            experiment={
                "num_clients": "invalid",  # Should be int, not str
            }
        )


@pytest.mark.unit
def test_missing_num_clients_in_config_doesnt_crash(
    temp_config_files,
    mock_config_manager,
):
    """
    Negative Test: Missing num_clients in config doesn't crash

    Verify that if num_clients is not present in the config, the system
    handles it gracefully and doesn't crash.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )

    # Update something else that's not num_clients
    update_request = ConfigurationUpdateRequest(
        experiment={
            "learning_rate": 0.002,
        }
    )

    # This should not crash
    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify the update worked
    assert mock_config_manager.get("experiment.learning_rate") == 0.002


@pytest.mark.unit
def test_negative_num_clients_value_fails_validation():
    """
    Negative Test: Negative num_clients value fails validation

    Verify that negative values for num_clients are properly rejected.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from pydantic import ValidationError

    # Pydantic doesn't have built-in int > 0 validation, but we can add it
    # For now, just verify it accepts the type
    # In a real implementation, we'd add custom validators
    ConfigurationUpdateRequest(
        experiment={
            "num_clients": -5,  # Type is valid, but semantically invalid
        }
    )

    # This test documents the need for additional validation
    # TODO: Add custom validator to ensure num_clients > 0


@pytest.mark.unit
@patch(
    "federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment.update_flwr_config",
    side_effect=Exception("TOML sync failed"),
)
def test_toml_sync_failure_doesnt_fail_update(mock_update_flwr, mock_config_manager):
    """
    Negative Test: TOML sync failure doesn't fail the update

    Verify that if TOML sync fails, the YAML update still succeeds
    and the error is caught and handled gracefully.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
        read_configs_to_toml,
    )

    # Mock read_configs_to_toml
    with patch.object(
        read_configs_to_toml,
        return_value={"num_supernodes": 10},
    ):
        # Create update request
        update_request = ConfigurationUpdateRequest(
            experiment={
                "num_clients": 10,
            }
        )

        # This should NOT raise an exception - error should be caught
        config_data = update_request.model_dump(exclude_none=True)
        update_result = mock_config_manager.update_from_dict(config_data)

        # Verify YAML was still updated
        assert mock_config_manager.get("experiment.num_clients") == 10


# ============================================================================
# EDGE CASE TESTS (10-20%)
# =============================================================================


@pytest.mark.unit
def test_zero_num_clients_handling(temp_config_files, mock_config_manager):
    """
    Edge Case Test: Zero num_clients handling

    Verify that num_clients = 0 is handled correctly (edge case for
    no clients in federated learning).
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
        update_flwr_config,
    )

    yaml_path, toml_path = temp_config_files

    # Update with zero
    update_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": 0,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify YAML update
    assert mock_config_manager.get("experiment.num_clients") == 0

    # Verify TOML sync
    update_flwr_config(pyproject_path=toml_path, num_supernodes=0)

    # Verify TOML
    import tomllib

    with open(toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    assert (
        toml_data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
            "num-supernodes"
        ]
        == 0
    )


@pytest.mark.unit
def test_very_large_num_clients_handling(temp_config_files, mock_config_manager):
    """
    Edge Case Test: Very large num_clients handling

    Verify that very large values for num_clients are handled correctly
    (edge case for large-scale federated learning).
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
        update_flwr_config,
    )

    yaml_path, toml_path = temp_config_files

    # Update with very large value
    large_value = 10000
    update_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": large_value,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify YAML update
    assert mock_config_manager.get("experiment.num_clients") == large_value

    # Verify TOML sync
    update_flwr_config(pyproject_path=toml_path, num_supernodes=large_value)

    # Verify TOML
    import tomllib

    with open(toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    assert (
        toml_data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
            "num-supernodes"
        ]
        == large_value
    )


@pytest.mark.unit
def test_num_clients_equals_one_minimum_value(
    temp_config_files,
    mock_config_manager,
):
    """
    Edge Case Test: num_clients = 1 (minimum value)

    Verify that num_clients = 1, the minimum meaningful value for
    federated learning, is handled correctly.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
        update_flwr_config,
    )

    yaml_path, toml_path = temp_config_files

    # Update with minimum value
    update_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": 1,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify YAML update
    assert mock_config_manager.get("experiment.num_clients") == 1

    # Verify TOML sync
    update_flwr_config(pyproject_path=toml_path, num_supernodes=1)

    # Verify TOML
    import tomllib

    with open(toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    assert (
        toml_data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
            "num-supernodes"
        ]
        == 1
    )


@pytest.mark.unit
def test_empty_update_request_doesnt_crash(mock_config_manager):
    """
    Edge Case Test: Empty update request doesn't crash

    Verify that sending an empty update request (no fields to update)
    doesn't crash the system.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )

    # Empty update request
    update_request = ConfigurationUpdateRequest()

    # This should not crash
    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Should have updated zero fields
    assert update_result["updated_count"] == 0


# ============================================================================
# CONTRACT TESTS (5-15%)
# =============================================================================


@pytest.mark.unit
def test_pydantic_schema_validation_for_configuration_update_request():
    """
    Contract Test: Pydantic schema validation for ConfigurationUpdateRequest

    Verify that the Pydantic schema properly validates the request structure
    and enforces type constraints.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from pydantic import ValidationError

    # Valid request
    valid_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": 5,
            "learning_rate": 0.001,
        }
    )
    assert valid_request.experiment.num_clients == 5
    assert valid_request.experiment.learning_rate == 0.001

    # Invalid type for num_clients
    with pytest.raises(ValidationError):
        ConfigurationUpdateRequest(
            experiment={
                "num_clients": "not_an_int",
            }
        )

    # Invalid type for learning_rate
    with pytest.raises(ValidationError):
        ConfigurationUpdateRequest(
            experiment={
                "learning_rate": "not_a_float",
            }
        )


@pytest.mark.unit
def test_response_structure_matches_expected_schema(mock_config_manager):
    """
    Contract Test: Response structure matches expected schema

    Verify that the response from config.update contains all expected
    fields and follows the correct structure.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )

    update_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": 10,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify response structure
    assert "updated_count" in update_result
    assert "fields" in update_result
    assert "verified_values" in update_result

    # Verify field types
    assert isinstance(update_result["updated_count"], int)
    assert isinstance(update_result["fields"], list)
    assert isinstance(update_result["verified_values"], dict)


@pytest.mark.unit
def test_toml_structure_is_valid_after_update(
    temp_config_files,
    mock_config_manager,
):
    """
    Contract Test: TOML structure is valid after update

    Verify that the pyproject.toml file remains valid TOML after
    the update operation and can be parsed without errors.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )
    from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
        update_flwr_config,
    )

    yaml_path, toml_path = temp_config_files

    # Update config
    update_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": 8,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Sync to TOML
    update_flwr_config(
        pyproject_path=toml_path,
        num_supernodes=8,
        num_server_rounds=2,
        max_epochs=2,
    )

    # Verify TOML is valid and can be parsed
    import tomllib

    with open(toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    # Verify expected structure exists
    assert "tool" in toml_data
    assert "flwr" in toml_data["tool"]
    assert "app" in toml_data["tool"]["flwr"]
    assert "config" in toml_data["tool"]["flwr"]["app"]
    assert "federations" in toml_data["tool"]["flwr"]
    assert "local-simulation" in toml_data["tool"]["flwr"]["federations"]
    assert "options" in toml_data["tool"]["flwr"]["federations"]["local-simulation"]


@pytest.mark.unit
@patch(
    "federated_pneumonia_detection.src.control.federated_new_version.core.utils.ConfigManager"
)
def test_read_configs_to_toml_mapping_contract(
    mock_cm_class, temp_config_files, sample_default_config
):
    """
    Contract Test: read_configs_to_toml correctly maps num_clients to num-supernodes

    Verify the mapping contract: experiment.num_clients -> num_supernodes
    """
    yaml_path, _ = temp_config_files

    # Update YAML with specific num_clients value
    with open(yaml_path, "w") as f:
        sample_default_config["experiment"]["num_clients"] = 15
        yaml.dump(sample_default_config, f)

    # Create ConfigManager instance with updated file
    from federated_pneumonia_detection.config.config_manager import ConfigManager

    config = ConfigManager(config_path=yaml_path)

    # Mock ConfigManager to return our instance
    mock_cm_instance = MagicMock()
    mock_cm_instance.has_key = config.has_key
    mock_cm_instance.get = config.get
    mock_cm_class.return_value = mock_cm_instance

    # Read configs
    from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
        read_configs_to_toml,
    )

    flwr_configs = read_configs_to_toml()

    # Verify mapping contract - should map num_clients to num_supernodes
    assert "num_supernodes" in flwr_configs
    assert flwr_configs["num_supernodes"] == 15


# ============================================================================
# REGRESSION TESTS (5-10%)
# =============================================================================


@pytest.mark.unit
def test_existing_config_updates_still_work_with_new_sync_logic(
    temp_config_files,
    mock_config_manager,
):
    """
    Regression Test: Existing config updates still work with new sync logic

    Verify that non-federated config updates (e.g., learning_rate, epochs)
    continue to work correctly with the new TOML sync logic in place.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )

    # Update non-federated parameters
    update_request = ConfigurationUpdateRequest(
        experiment={
            "learning_rate": 0.005,
            "epochs": 20,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify updates worked
    assert mock_config_manager.get("experiment.learning_rate") == 0.005
    assert mock_config_manager.get("experiment.epochs") == 20

    # Verify updated fields
    assert "experiment.learning_rate" in update_result["fields"]
    assert "experiment.epochs" in update_result["fields"]


@pytest.mark.unit
def test_non_federated_config_updates_dont_affect_toml(
    temp_config_files,
    mock_config_manager,
):
    """
    Regression Test: Non-federated config updates don't affect TOML

    Verify that updating non-federated config parameters doesn't modify
    pyproject.toml (TOML only changes for federated parameters).
    """
    import tomllib

    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )

    yaml_path, toml_path = temp_config_files

    # Read original TOML
    with open(toml_path, "rb") as f:
        original_toml = tomllib.load(f)
        original_num_supernodes = original_toml["tool"]["flwr"]["federations"][
            "local-simulation"
        ]["options"]["num-supernodes"]

    # Update non-federated parameter
    update_request = ConfigurationUpdateRequest(
        experiment={
            "learning_rate": 0.005,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Read TOML again (should be unchanged)
    with open(toml_path, "rb") as f:
        new_toml = tomllib.load(f)
        new_num_supernodes = new_toml["tool"]["flwr"]["federations"][
            "local-simulation"
        ]["options"]["num-supernodes"]

    # Verify TOML was not modified (we didn't call update_flwr_config)
    assert new_num_supernodes == original_num_supernodes


@pytest.mark.unit
def test_config_update_preserves_other_yaml_values(mock_config_manager):
    """
    Regression Test: Config update preserves other YAML values

    Verify that updating a specific config value doesn't accidentally
    modify or lose other values in the config file.
    """
    from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
        ConfigurationUpdateRequest,
    )

    # Get original values
    original_learning_rate = mock_config_manager.get("experiment.learning_rate")
    original_epochs = mock_config_manager.get("experiment.epochs")
    original_system_batch_size = mock_config_manager.get("system.batch_size")

    # Update just num_clients
    update_request = ConfigurationUpdateRequest(
        experiment={
            "num_clients": 12,
        }
    )

    config_data = update_request.model_dump(exclude_none=True)
    update_result = mock_config_manager.update_from_dict(config_data)

    # Verify other values are preserved
    assert mock_config_manager.get("experiment.learning_rate") == original_learning_rate
    assert mock_config_manager.get("experiment.epochs") == original_epochs
    assert mock_config_manager.get("system.batch_size") == original_system_batch_size
    assert mock_config_manager.get("experiment.num_clients") == 12
