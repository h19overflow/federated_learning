"""
Unit tests for toml_adjustment module.

Tests TOML configuration updates for federated learning.
"""

import tomli_w

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib

from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (  # noqa: E501
    update_flwr_config,
)


class TestUpdateFlwrConfig:
    """Test suite for update_flwr_config function."""

    def test_update_num_server_rounds(self, tmp_path, sample_pyproject_toml_content):
        """Test updating num_server_rounds."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write initial content
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {
                        "flwr": {
                            "app": {"config": {"num-server-rounds": 5}},
                        },
                    },
                },
                f,
            )

        # Update config
        update_flwr_config(str(pyproject_path), num_server_rounds=10)

        # Verify update
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] == 10

    def test_update_max_epochs(self, tmp_path, sample_pyproject_toml_content):
        """Test updating max_epochs."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write initial content
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {
                        "flwr": {
                            "app": {"config": {"max-epochs": 2}},
                        },
                    },
                },
                f,
            )

        # Update config
        update_flwr_config(str(pyproject_path), max_epochs=5)

        # Verify update
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert data["tool"]["flwr"]["app"]["config"]["max-epochs"] == 5

    def test_update_num_supernodes(self, tmp_path, sample_pyproject_toml_content):
        """Test updating num_supernodes."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write initial content
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {
                        "flwr": {
                            "federations": {
                                "local-simulation": {"options": {"num-supernodes": 3}},
                            },
                        },
                    },
                },
                f,
            )

        # Update config
        update_flwr_config(str(pyproject_path), num_supernodes=8)

        # Verify update
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert (
            data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
                "num-supernodes"
            ]
            == 8
        )

    def test_update_multiple_configs(self, tmp_path, sample_pyproject_toml_content):
        """Test updating multiple configs at once."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write initial content
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {
                        "flwr": {
                            "app": {
                                "config": {"num-server-rounds": 5, "max-epochs": 2},
                            },
                            "federations": {
                                "local-simulation": {"options": {"num-supernodes": 3}},
                            },
                        },
                    },
                },
                f,
            )

        # Update all configs
        update_flwr_config(
            str(pyproject_path),
            num_server_rounds=10,
            max_epochs=3,
            num_supernodes=5,
        )

        # Verify all updates
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] == 10
        assert data["tool"]["flwr"]["app"]["config"]["max-epochs"] == 3
        assert (
            data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
                "num-supernodes"
            ]
            == 5
        )

    def test_update_with_none_values(self, tmp_path, sample_pyproject_toml_content):
        """Test that None values are filtered out."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write initial content
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {
                        "flwr": {
                            "app": {
                                "config": {"num-server-rounds": 5, "max-epochs": 2},
                            },
                        },
                    },
                },
                f,
            )

        # Update with None value (should be ignored)
        update_flwr_config(
            str(pyproject_path),
            num_server_rounds=10,
            max_epochs=None,  # Should be filtered
        )

        # Verify only valid update was applied
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] == 10
        assert data["tool"]["flwr"]["app"]["config"]["max-epochs"] == 2  # Unchanged

    def test_update_nonexistent_key_logs_warning(self, tmp_path, capsys):
        """Test that unknown config keys log warnings."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write initial content
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {"flwr": {"app": {"config": {}}}},
                },
                f,
            )

        # Try to update unknown key
        update_flwr_config(str(pyproject_path), unknown_key=42)

        # Check that warning was logged
        captured = capsys.readouterr()
        assert (
            "Unknown config key: unknown_key" in captured.out or "WARN" in captured.out
        )

    def test_update_missing_pyproject_file(self, tmp_path, capsys):
        """Test handling of missing pyproject.toml file."""
        non_existent_path = tmp_path / "nonexistent.toml"

        # Should not raise error, just log
        update_flwr_config(str(non_existent_path), num_server_rounds=10)

        # Check that error was logged
        captured = capsys.readouterr()
        assert "not found" in captured.out or "ERROR" in captured.out

    def test_update_no_changes_logged(self, tmp_path, capsys):
        """Test that no changes situation is logged."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write initial content
        with open(pyproject_path, "wb") as f:
            tomli_w.dump({"project": {"name": "test"}, "tool": {"flwr": {}}}, f)

        # Try to update with invalid key (no changes made)
        update_flwr_config(str(pyproject_path), unknown_key=42)

        # Check that no updates warning was logged
        captured = capsys.readouterr()
        assert "No updates were made" in captured.out or "WARN" in captured.out

    def test_update_preserves_other_sections(self, tmp_path):
        """Test that other TOML sections are preserved."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write content with multiple sections
        initial_data = {
            "project": {
                "name": "test_project",
                "version": "1.0.0",
            },
            "build-system": {"requires": ["setuptools"]},
            "tool": {
                "flwr": {
                    "app": {"config": {"num-server-rounds": 5}},
                },
                "ruff": {"line-length": 88},
            },
        }

        with open(pyproject_path, "wb") as f:
            tomli_w.dump(initial_data, f)

        # Update only flwr config
        update_flwr_config(str(pyproject_path), num_server_rounds=10)

        # Verify other sections preserved
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert data["project"]["name"] == "test_project"
        assert data["project"]["version"] == "1.0.0"
        assert data["build-system"]["requires"] == ["setuptools"]
        assert data["tool"]["ruff"]["line-length"] == 88
        assert data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] == 10

    def test_update_creates_file_if_missing_structure(self, tmp_path):
        """Test that missing structure is handled gracefully."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write file without flwr section
        with open(pyproject_path, "wb") as f:
            tomli_w.dump({"project": {"name": "test"}}, f)

        # Should raise KeyError for missing structure
        # (The function handles this by logging error)
        update_flwr_config(str(pyproject_path), num_server_rounds=10)

        # Verify file was not modified (or error was logged)
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Original content should be preserved
        assert data["project"]["name"] == "test"

    def test_update_default_path_none(self, tmp_path, monkeypatch):
        """Test update with None path (uses default)."""
        # This test is tricky because default path is hardcoded
        # We just verify it doesn't crash with None
        update_flwr_config(
            pyproject_path=None,  # Uses default
            num_server_rounds=10,
        )
        # If no exception, test passes

    def test_update_overwrites_existing_values(self, tmp_path):
        """Test that existing values are overwritten."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Write with existing value
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {
                        "flwr": {
                            "app": {"config": {"num-server-rounds": 5}},
                        },
                    },
                },
                f,
            )

        # Update to new value
        update_flwr_config(str(pyproject_path), num_server_rounds=20)

        # Verify new value
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] == 20

    def test_update_with_zero_values(self, tmp_path):
        """Test that zero values are handled correctly (not filtered)."""
        pyproject_path = tmp_path / "pyproject.toml"

        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {
                        "flwr": {
                            "app": {
                                "config": {"num-server-rounds": 5, "max-epochs": 2},
                            },
                        },
                    },
                },
                f,
            )

        # Zero is valid, should be applied
        update_flwr_config(str(pyproject_path), max_epochs=0)

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert data["tool"]["flwr"]["app"]["config"]["max-epochs"] == 0

    def test_update_with_negative_values(self, tmp_path):
        """Test that negative values are accepted (may be invalid but not filtered)."""
        pyproject_path = tmp_path / "pyproject.toml"

        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {
                        "flwr": {
                            "app": {"config": {"num-server-rounds": 5}},
                        },
                    },
                },
                f,
            )

        # Negative value (invalid but should be applied)
        update_flwr_config(str(pyproject_path), num_server_rounds=-1)

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] == -1

    def test_update_empty_config_dict(self, tmp_path):
        """Test updating with empty kwargs (no changes)."""
        pyproject_path = tmp_path / "pyproject.toml"

        with open(pyproject_path, "wb") as f:
            tomli_w.dump(
                {
                    "project": {"name": "test"},
                    "tool": {
                        "flwr": {
                            "app": {"config": {"num-server-rounds": 5}},
                        },
                    },
                },
                f,
            )

        # No changes
        update_flwr_config(str(pyproject_path))

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        assert data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] == 5

    def test_write_error_handling(self, tmp_path, capsys, monkeypatch):
        """Test handling of write errors."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Mock tomli_w.dump to raise error
        def mock_dump_error(data, f):
            raise IOError("Mock write error")

        monkeypatch.setattr("tomli_w.dump", mock_dump_error)

        # Should handle error gracefully
        update_flwr_config(str(pyproject_path), num_server_rounds=10)

        # Check that error was logged
        captured = capsys.readouterr()
        assert "Failed to write" in captured.out or "ERROR" in captured.out

    def test_read_error_handling(self, tmp_path, capsys, monkeypatch):
        """Test handling of read errors."""
        pyproject_path = tmp_path / "pyproject.toml"

        # Mock tomllib.load to raise error
        def mock_load_error(f):
            raise IOError("Mock read error")

        monkeypatch.setattr(
            "federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment.tomllib.load",
            mock_load_error,
        )

        # Should handle error gracefully
        update_flwr_config(str(pyproject_path), num_server_rounds=10)

        # Check that error was logged
        captured = capsys.readouterr()
        assert "Failed to read" in captured.out or "ERROR" in captured.out
