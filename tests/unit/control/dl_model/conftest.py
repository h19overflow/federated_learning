from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image


@pytest.fixture
def mock_config():
    """Mock ConfigManager for testing."""
    config = MagicMock()
    # Default return values for common config keys
    config_values = {
        "experiment.learning_rate": 0.001,
        "experiment.weight_decay": 0.0001,
        "experiment.dropout_rate": 0.5,
        "experiment.fine_tune_layers_count": 0,
        "experiment.use_torch_compile": False,
        "experiment.epochs": 10,
        "experiment.min_lr": 1e-7,
        "experiment.reduce_lr_factor": 0.5,
        "experiment.reduce_lr_patience": 3,
    }

    def get_side_effect(key, default=None):
        return config_values.get(key, default)

    config.get.side_effect = get_side_effect
    return config


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for dataset testing."""
    data = {
        "path": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        "label": [0, 1, 0, 1],
        "split": ["train", "train", "val", "val"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_image_dir(tmp_path):
    """Create a temporary directory with dummy images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create dummy images
    for i in range(1, 5):
        img_path = img_dir / f"img{i}.jpg"
        # Create a random RGB image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_path)

    return img_dir


@pytest.fixture
def mock_transform():
    """Mock transform that returns a tensor."""
    return MagicMock(return_value=torch.randn(3, 224, 224))
