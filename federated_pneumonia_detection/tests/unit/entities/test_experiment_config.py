"""
Unit tests for ExperimentConfig entity class.
Tests configuration validation, parameter handling, and serialization.
"""

import pytest
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.models.system_constants import SystemConstants


class TestExperimentConfig:
    """Test cases for ExperimentConfig entity."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ExperimentConfig()

        # Model parameters
        assert config.learning_rate == 0.001
        assert config.epochs == 15
        assert config.weight_decay == 0.0001
        assert config.freeze_backbone is True

        # Data parameters
        assert config.sample_fraction == 0.10
        assert config.validation_split == 0.20
        assert config.batch_size == 512

        # Training parameters
        assert config.early_stopping_patience == 5
        assert config.reduce_lr_patience == 3
        assert config.reduce_lr_factor == 0.5
        assert config.min_lr == 1e-7

        # Federated Learning parameters
        assert config.num_rounds == 2
        assert config.num_clients == 2
        assert config.clients_per_round == 2
        assert config.local_epochs == 15

        # System parameters
        assert config.seed == 42
        assert config.device == 'auto'
        assert config.num_workers == 4

    def test_parameter_validation_positive_learning_rate(self):
        """Test that learning rate must be positive."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            ExperimentConfig(learning_rate=0)

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            ExperimentConfig(learning_rate=-0.001)

    def test_parameter_validation_positive_epochs(self):
        """Test that epochs must be positive."""
        with pytest.raises(ValueError, match="Number of epochs must be positive"):
            ExperimentConfig(epochs=0)

        with pytest.raises(ValueError, match="Number of epochs must be positive"):
            ExperimentConfig(epochs=-5)

    def test_parameter_validation_sample_fraction_range(self):
        """Test that sample fraction is in valid range."""
        with pytest.raises(ValueError, match="Sample fraction must be between 0 and 1"):
            ExperimentConfig(sample_fraction=0)

        with pytest.raises(ValueError, match="Sample fraction must be between 0 and 1"):
            ExperimentConfig(sample_fraction=1.5)

        with pytest.raises(ValueError, match="Sample fraction must be between 0 and 1"):
            ExperimentConfig(sample_fraction=-0.1)

        # Valid values should work
        config = ExperimentConfig(sample_fraction=0.5)
        assert config.sample_fraction == 0.5

        config = ExperimentConfig(sample_fraction=1.0)
        assert config.sample_fraction == 1.0

    def test_parameter_validation_validation_split_range(self):
        """Test that validation split is in valid range."""
        with pytest.raises(ValueError, match="Validation split must be between 0 and 1"):
            ExperimentConfig(validation_split=0)

        with pytest.raises(ValueError, match="Validation split must be between 0 and 1"):
            ExperimentConfig(validation_split=1.0)

        with pytest.raises(ValueError, match="Validation split must be between 0 and 1"):
            ExperimentConfig(validation_split=1.5)

        # Valid values should work
        config = ExperimentConfig(validation_split=0.3)
        assert config.validation_split == 0.3

    def test_parameter_validation_positive_batch_size(self):
        """Test that batch size must be positive."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            ExperimentConfig(batch_size=0)

        with pytest.raises(ValueError, match="Batch size must be positive"):
            ExperimentConfig(batch_size=-32)

    def test_parameter_validation_fl_parameters(self):
        """Test federated learning parameter validation."""
        with pytest.raises(ValueError, match="Number of FL rounds must be positive"):
            ExperimentConfig(num_rounds=0)

        with pytest.raises(ValueError, match="Number of clients must be positive"):
            ExperimentConfig(num_clients=0)

        with pytest.raises(ValueError, match="Clients per round cannot exceed total number of clients"):
            ExperimentConfig(num_clients=3, clients_per_round=5)

    def test_from_system_constants(self):
        """Test creating config from system constants."""
        constants = SystemConstants.create_custom(
            batch_size=64,
            sample_fraction=0.25,
            seed=999
        )

        config = ExperimentConfig.from_system_constants(constants)

        assert config.batch_size == 64
        assert config.sample_fraction == 0.25
        assert config.seed == 999
        # Other values should be defaults
        assert config.learning_rate == 0.001
        assert config.epochs == 15

    def test_from_system_constants_with_overrides(self):
        """Test creating config from system constants with additional parameters."""
        constants = SystemConstants()

        config = ExperimentConfig.from_system_constants(
            constants,
            learning_rate=0.01,
            epochs=20,
            freeze_backbone=False
        )

        # Values from constants (SystemConstants defaults)
        assert config.batch_size == 128
        assert config.sample_fraction == 0.05
        assert config.seed == 42

        # Overridden values
        assert config.learning_rate == 0.01
        assert config.epochs == 20
        assert config.freeze_backbone is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ExperimentConfig(
            learning_rate=0.01,
            epochs=20,
            batch_size=64
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['learning_rate'] == 0.01
        assert config_dict['epochs'] == 20
        assert config_dict['batch_size'] == 64
        assert 'metadata' in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'learning_rate': 0.005,
            'epochs': 15,
            'batch_size': 256,
            'freeze_backbone': False,
            'seed': 999
        }

        config = ExperimentConfig.from_dict(config_dict)

        assert config.learning_rate == 0.005
        assert config.epochs == 15
        assert config.batch_size == 256
        assert config.freeze_backbone is False
        assert config.seed == 999

    def test_metadata_field(self):
        """Test metadata field functionality."""
        config = ExperimentConfig()
        assert config.metadata == {}

        # Test with custom metadata
        custom_metadata = {'experiment_name': 'test_run', 'version': '1.0'}
        config_with_metadata = ExperimentConfig(metadata=custom_metadata)
        assert config_with_metadata.metadata == custom_metadata

    def test_round_trip_serialization(self):
        """Test that to_dict -> from_dict preserves all values."""
        original_config = ExperimentConfig(
            learning_rate=0.002,
            epochs=25,
            batch_size=32,
            metadata={'test': True}
        )

        config_dict = original_config.to_dict()
        reconstructed_config = ExperimentConfig.from_dict(config_dict)

        assert original_config.learning_rate == reconstructed_config.learning_rate
        assert original_config.epochs == reconstructed_config.epochs
        assert original_config.batch_size == reconstructed_config.batch_size
        assert original_config.metadata == reconstructed_config.metadata