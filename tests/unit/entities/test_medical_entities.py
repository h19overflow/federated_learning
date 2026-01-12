"""
Unit tests for medical entities (CustomImageDataset and ResNetWithCustomHead).
Focuses on achieving 100% code coverage through exhaustive testing of all methods and edge cases.
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import torch.nn as nn
from unittest.mock import MagicMock, patch

from federated_pneumonia_detection.src.entities.custom_image_dataset import CustomImageDataset
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead
from tests.fixtures.sample_data import SampleDataFactory


class TestCustomImageDataset:
    """Test suite for CustomImageDataset."""

    @pytest.fixture
    def valid_df(self):
        return pd.DataFrame({
            'filename': ['sample.png'],
            'Target': [1]
        })

    def test_init_success(self, temp_data_structure, valid_df):
        """Test successful initialization with valid data."""
        # Create the file mentioned in valid_df
        SampleDataFactory.create_dummy_image().save(Path(temp_data_structure['base_path']) / 'sample.png')
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path'],
            validate_images=True
        )
        assert len(dataset) == 1
        assert dataset.color_mode == "RGB"

    def test_init_with_config(self, temp_data_structure, sample_config, valid_df):
        """Test initialization with explicit ConfigManager."""
        SampleDataFactory.create_dummy_image().save(Path(temp_data_structure['base_path']) / 'sample.png')
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path'],
            config=sample_config
        )
        assert dataset.config == sample_config

    def test_init_empty_dataframe(self, temp_data_structure):
        """Test initialization with an empty dataframe."""
        dataset = CustomImageDataset(
            dataframe=pd.DataFrame(columns=['filename', 'Target']),
            image_dir=temp_data_structure['base_path']
        )
        assert len(dataset) == 0
        assert dataset.filenames.size == 0

    def test_init_invalid_dataframe_type(self, temp_data_structure):
        """Test initialization with invalid dataframe type."""
        with pytest.raises(ValueError, match="dataframe must be a pandas DataFrame"):
            CustomImageDataset(dataframe="not a dataframe", image_dir=temp_data_structure['base_path'])

    def test_init_missing_image_dir(self):
        """Test initialization with non-existent image directory."""
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            CustomImageDataset(dataframe=pd.DataFrame(), image_dir="/non/existent/path")

    def test_init_path_is_not_dir(self, tmp_path):
        """Test initialization where image_dir is a file instead of a directory."""
        file_path = tmp_path / "test.txt"
        file_path.touch()
        with pytest.raises(ValueError, match="Image directory path is not a directory"):
            CustomImageDataset(dataframe=pd.DataFrame(), image_dir=file_path)

    def test_init_invalid_color_mode(self, temp_data_structure, valid_df):
        """Test initialization with unsupported color mode."""
        with pytest.raises(ValueError, match="Color mode must be 'RGB' or 'L'"):
            CustomImageDataset(dataframe=valid_df, image_dir=temp_data_structure['base_path'], color_mode="CMYK")

    def test_init_missing_columns(self, temp_data_structure):
        """Test initialization with missing required columns in dataframe."""
        df = pd.DataFrame({'wrong_col': [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            CustomImageDataset(dataframe=df, image_dir=temp_data_structure['base_path'])

    def test_validate_image_files_failure(self, temp_data_structure, valid_df):
        """Test image validation when some files are missing or invalid."""
        # Add a missing file and an invalid image (not a real image)
        df = pd.DataFrame({
            'filename': ['missing.png', 'invalid.png', 'valid.png'],
            'Target': [0, 1, 0]
        })
        base_path = Path(temp_data_structure['base_path'])
        (base_path / 'invalid.png').write_text("not an image")
        # Create one valid image
        SampleDataFactory.create_dummy_image().save(base_path / 'valid.png')
        
        dataset = CustomImageDataset(
            dataframe=df,
            image_dir=temp_data_structure['base_path'],
            validate_images=True
        )
        assert len(dataset) == 1
        assert dataset.filenames[dataset.valid_indices[0]] == 'valid.png'

    def test_getitem_success(self, temp_data_structure, valid_df):
        """Test successful __getitem__ execution."""
        base_path = Path(temp_data_structure['base_path'])
        SampleDataFactory.create_dummy_image().save(base_path / 'sample.png')
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path'],
            validate_images=True
        )
        img, label = dataset[0]
        assert isinstance(img, Image.Image)
        assert float(label) == 1.0

    def test_getitem_with_transform(self, temp_data_structure, valid_df):
        """Test __getitem__ with a transform pipeline."""
        base_path = Path(temp_data_structure['base_path'])
        SampleDataFactory.create_dummy_image().save(base_path / 'sample.png')
        
        def mock_transform(img):
            return torch.zeros((3, 224, 224))

        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path'],
            transform=mock_transform
        )
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)

    def test_getitem_index_error(self, temp_data_structure, valid_df):
        """Test __getitem__ with out-of-bounds index."""
        SampleDataFactory.create_dummy_image().save(Path(temp_data_structure['base_path']) / 'sample.png')
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path']
        )
        with pytest.raises(IndexError):
            dataset[999]
        with pytest.raises(IndexError):
            dataset[-1]

    def test_getitem_runtime_error(self, temp_data_structure, valid_df):
        """Test __getitem__ when image loading fails."""
        SampleDataFactory.create_dummy_image().save(Path(temp_data_structure['base_path']) / 'sample.png')
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path']
        )
        
        with patch.object(CustomImageDataset, '_load_image', side_effect=Exception("Load failed")):
            with pytest.raises(RuntimeError, match="Failed to load sample"):
                dataset[0]

    def test_load_image_grayscale(self, temp_data_structure, valid_df):
        """Test loading image in grayscale mode."""
        base_path = Path(temp_data_structure['base_path'])
        SampleDataFactory.create_dummy_image().save(base_path / 'sample.png')
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path'],
            color_mode="L"
        )
        img = dataset._load_image('sample.png')
        assert img.mode == "L"

    def test_load_image_failure(self, temp_data_structure):
        """Test _load_image failure handling."""
        dataset = CustomImageDataset(
            dataframe=pd.DataFrame(),
            image_dir=temp_data_structure['base_path']
        )
        with pytest.raises(RuntimeError, match="Failed to load image"):
            dataset._load_image("non_existent.png")

    def test_get_class_distribution(self, temp_data_structure):
        """Test calculation of class distribution."""
        df = pd.DataFrame({
            'filename': ['a.png', 'b.png'],
            'Target': [0, 1]
        })
        base_path = Path(temp_data_structure['base_path'])
        (base_path / 'a.png').touch()
        (base_path / 'b.png').touch()

        dataset = CustomImageDataset(
            dataframe=df,
            image_dir=temp_data_structure['base_path'],
            validate_images=False
        )
        dist = dataset.get_class_distribution()
        assert dist == {0: 1, 1: 1}

    def test_get_class_distribution_empty(self, temp_data_structure):
        """Test class distribution for empty dataset."""
        dataset = CustomImageDataset(pd.DataFrame(), temp_data_structure['base_path'])
        assert dataset.get_class_distribution() == {}

    def test_get_sample_info(self, temp_data_structure, valid_df):
        """Test retrieval of detailed sample info."""
        base_path = Path(temp_data_structure['base_path'])
        SampleDataFactory.create_dummy_image().save(base_path / 'sample.png')
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path']
        )
        info = dataset.get_sample_info(0)
        assert info['index'] == 0
        assert 'image_size' in info
        
        with pytest.raises(IndexError):
            dataset.get_sample_info(999)

    def test_get_sample_info_invalid_image(self, temp_data_structure, valid_df):
        """Test get_sample_info for an invalid image file."""
        base_path = Path(temp_data_structure['base_path'])
        (base_path / 'sample.png').write_text("invalid")
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path'],
            validate_images=False
        )
        info = dataset.get_sample_info(0)
        assert "image_error" in info

    def test_validate_all_images(self, temp_data_structure, valid_df):
        """Test manual validation of all images including failures."""
        df = pd.DataFrame({
            'filename': ['valid.png', 'missing.png', 'invalid.png'],
            'Target': [0, 1, 0]
        })
        base_path = Path(temp_data_structure['base_path'])
        SampleDataFactory.create_dummy_image().save(base_path / 'valid.png')
        (base_path / 'invalid.png').write_text("corrupt")
        
        dataset = CustomImageDataset(
            dataframe=df,
            image_dir=temp_data_structure['base_path'],
            validate_images=False
        )
        valid, invalid, details = dataset.validate_all_images()
        assert valid == 1
        assert invalid == 2

    def test_get_memory_usage_estimate(self, temp_data_structure, valid_df):
        """Test memory usage estimation logic."""
        base_path = Path(temp_data_structure['base_path'])
        SampleDataFactory.create_dummy_image().save(base_path / 'sample.png')
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path']
        )
        est = dataset.get_memory_usage_estimate()
        assert est['total_samples'] == 1
        assert est['estimated_total_memory_mb'] > 0

    def test_get_memory_usage_estimate_fail_sample(self, temp_data_structure, valid_df):
        """Test memory usage estimation when some samples fail to load."""
        base_path = Path(temp_data_structure['base_path'])
        (base_path / 'sample.png').write_text("corrupt")
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path'],
            validate_images=False
        )
        est = dataset.get_memory_usage_estimate()
        assert est['total_samples'] == 1
        assert est['avg_pixels_per_image'] == 0
        assert est['estimated_total_memory_mb'] == 0

    def test_get_memory_usage_estimate_all_sample_fail(self, temp_data_structure, valid_df):
        """Test memory usage estimation when ALL sampled images fail to open."""
        base_path = Path(temp_data_structure['base_path'])
        SampleDataFactory.create_dummy_image().save(base_path / 'sample.png')
        dataset = CustomImageDataset(
            dataframe=valid_df,
            image_dir=temp_data_structure['base_path'],
            validate_images=False
        )
        # Force Image.open to fail during sampling
        with patch('PIL.Image.open', side_effect=Exception("Sampling failed")):
            est = dataset.get_memory_usage_estimate()
            assert est['estimated_total_memory_mb'] == 0
            assert est['avg_pixels_per_image'] == 0

    def test_get_memory_usage_estimate_empty(self, temp_data_structure):
        """Test memory usage estimation for empty dataset."""
        dataset = CustomImageDataset(pd.DataFrame(), temp_data_structure['base_path'])
        est = dataset.get_memory_usage_estimate()
        assert est['total_samples'] == 0


class TestResNetWithCustomHead:
    """Test suite for ResNetWithCustomHead."""

    def test_init_success(self):
        """Test successful initialization with default parameters."""
        model = ResNetWithCustomHead(num_classes=1)
        assert isinstance(model.features, nn.Sequential)
        assert isinstance(model.classifier, nn.Sequential)
        assert model.num_classes == 1

    def test_init_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            ResNetWithCustomHead(num_classes=0)
        
        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
            ResNetWithCustomHead(dropout_rate=1.5)
            
        with pytest.raises(ValueError, match="fine_tune_layers_count must be an integer"):
            ResNetWithCustomHead(fine_tune_layers_count="none")

    def test_custom_head_sizes(self):
        """Test initialization with custom classifier head sizes."""
        custom_sizes = [2048, 128, 1]
        model = ResNetWithCustomHead(custom_head_sizes=custom_sizes)
        # Check linear layers in classifier
        linears = [m for m in model.classifier.modules() if isinstance(m, nn.Linear)]
        assert linears[0].out_features == 128
        assert linears[1].out_features == 1

    def test_custom_head_sizes_invalid(self):
        """Test initialization with too few custom head sizes."""
        with pytest.raises(ValueError, match="custom_head_sizes must have at least 2 elements"):
            ResNetWithCustomHead(custom_head_sizes=[2048])

    def test_forward_pass(self):
        """Test model forward pass with dummy input."""
        model = ResNetWithCustomHead(num_classes=1)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 1)

    def test_forward_pass_failure(self):
        """Test forward pass error handling."""
        model = ResNetWithCustomHead()
        with pytest.raises(RuntimeError, match="Forward pass failed"):
            model(torch.randn(1, 1, 10, 10)) # Wrong input size/channels

    def test_fine_tuning_unfreeze(self):
        """Test unfreezing logic for fine-tuning."""
        # Unfreeze last 2 layers
        model = ResNetWithCustomHead(fine_tune_layers_count=-2)
        trainable = [p for p in model.features.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_unfreeze_last_n_layers_all(self):
        """Test unfreezing all layers by providing a large N."""
        model = ResNetWithCustomHead()
        # Count parameter-containing layers
        param_layers = []
        for module in list(model.features.modules()):
            if any(param.numel() > 0 for param in module.parameters(recurse=False)):
                param_layers.append(module)
        
        model._unfreeze_last_n_layers(len(param_layers) + 10)
        assert all(p.requires_grad for p in model.features.parameters())

    def test_freeze_unfreeze_backbone(self):
        """Test manual freezing and unfreezing of the backbone."""
        model = ResNetWithCustomHead()
        model.unfreeze_backbone()
        assert all(p.requires_grad for p in model.features.parameters())
        
        model.freeze_backbone()
        assert not any(p.requires_grad for p in model.features.parameters())

    def test_set_dropout_rate(self):
        """Test updating the dropout rate dynamically."""
        model = ResNetWithCustomHead(dropout_rate=0.5)
        model.set_dropout_rate(0.2)
        assert model.dropout_rate == 0.2
        for m in model.classifier.modules():
            if isinstance(m, nn.Dropout):
                assert m.p == 0.2
        
        with pytest.raises(ValueError):
            model.set_dropout_rate(1.1)

    def test_get_model_info(self):
        """Test retrieval of model metadata."""
        model = ResNetWithCustomHead()
        info = model.get_model_info()
        assert info['model_name'] == "ResNetWithCustomHead"
        assert 'total_parameters' in info

    def test_get_feature_maps(self):
        """Test extraction of feature maps from specific layers."""
        model = ResNetWithCustomHead()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Test default (final features)
        fm = model.get_feature_maps(dummy_input)
        assert fm.shape[1] == 2048
        
        # Test specific layer
        # ResNet50 has layers named '0', '1', ... '7' in its sequential features
        fm_layer = model.get_feature_maps(dummy_input, layer_name='0')
        assert fm_layer is not None
        
        with pytest.raises(ValueError, match="not found"):
            model.get_feature_maps(dummy_input, layer_name="invalid_layer")

    def test_backbone_creation_failure(self):
        """Test error handling when backbone creation fails."""
        with patch('torchvision.models.resnet50', side_effect=Exception("Model load failed")):
            with pytest.raises(RuntimeError, match="Failed to create ResNet50 backbone"):
                ResNetWithCustomHead()

    def test_get_feature_maps_runtime_error(self):
        """Test RuntimeError when feature extraction hook fails to capture data."""
        model = ResNetWithCustomHead()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # We need to simulate the case where features dict is not populated
        # This can happen if the hook is registered but not called correctly
        with patch.object(nn.Module, 'register_forward_hook') as mock_hook:
            # mock_hook returns a handle that can be removed
            mock_handle = MagicMock()
            mock_hook.return_value = mock_handle
            
            with pytest.raises(RuntimeError, match="Failed to extract features"):
                # We target '0' which is a layer in ResNet backbone
                model.get_feature_maps(dummy_input, layer_name='0')
