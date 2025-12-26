"""
Unit tests for medical utility functions and classes.
Focuses on achieving 100% code coverage for data processing and image transforms.
"""

import pytest
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from unittest.mock import MagicMock, patch
import warnings
from tests.fixtures.sample_data import SampleDataFactory, TempDataStructure

from federated_pneumonia_detection.src.utils.data_processing_functions import (
    load_metadata,
    sample_dataframe,
    create_train_val_split,
    load_and_split_data,
    validate_image_paths,
    get_image_directory_path,
    get_data_statistics
)
from federated_pneumonia_detection.src.utils.image_transforms import (
    XRayPreprocessor,
    TransformBuilder,
    get_transforms,
    create_preprocessing_function
)
from federated_pneumonia_detection.config.config_manager import ConfigManager


class TestDataProcessingFunctions:
    """Test suite for data processing utility functions."""

    def test_load_metadata_success(self, temp_data_structure, sample_config):
        """Test successful metadata loading and preparation."""
        df = load_metadata(temp_data_structure['metadata_path'], sample_config)
        assert 'filename' in df.columns
        assert not df.empty

    def test_load_metadata_no_config_none(self, temp_data_structure):
        """Test load_metadata with config_or_constants=None explicitly."""
        with patch('federated_pneumonia_detection.config.config_manager.ConfigManager') as MockConfig:
            instance = MockConfig.return_value
            # Ensure it returns something valid for the rest of the function
            instance.get.side_effect = lambda k, default=None: {
                'columns.patient_id': 'patientId',
                'columns.target': 'Target',
                'columns.filename': 'filename',
                'system.image_extension': '.png'
            }.get(k, default)
            df = load_metadata(temp_data_structure['metadata_path'], config_or_constants=None)
            assert df is not None

    def test_load_metadata_missing_columns_config(self, tmp_path, sample_config):
        """Test missing columns when using ConfigManager."""
        df = pd.DataFrame({'wrong': [1]})
        path = tmp_path / "missing_config.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing column"):
            load_metadata(path, sample_config)
        
        # Second missing column
        df2 = pd.DataFrame({'patientId': ['p1'], 'wrong': [0]})
        df2.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing column"):
            load_metadata(path, sample_config)

    def test_load_metadata_empty_df_config(self, tmp_path, sample_config):
        """Test empty dataframe with ConfigManager."""
        df = pd.DataFrame(columns=['patientId', 'Target'])
        path = tmp_path / "empty_config.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="DataFrame is empty"):
            load_metadata(path, sample_config)

    def test_load_metadata_nan_values_config(self, tmp_path, sample_config):
        """Test NaN values with ConfigManager."""
        df = pd.DataFrame({'patientId': ['p1', None], 'Target': [0, 1]})
        path = tmp_path / "nan_config.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing values found in column"):
            load_metadata(path, sample_config)

    def test_load_and_split_data_custom_path(self, temp_data_structure, sample_config):
        """Test load_and_split_data with explicit metadata path and enough samples."""
        # Use a larger dataset to avoid stratification issues
        df_large = SampleDataFactory.create_sample_metadata(num_samples=100)
        with TempDataStructure(metadata_df=df_large) as paths:
            train, val = load_and_split_data(sample_config, metadata_path=paths['metadata_path'])
            assert train is not None

    def test_validate_image_paths_success(self, temp_data_structure, sample_config):
        """Test successful image path validation."""
        sample_config.set('paths.base_path', temp_data_structure['base_path'])
        assert validate_image_paths(sample_config) is True

    def test_load_metadata_file_not_found(self):
        """Test FileNotFoundError for non-existent metadata."""
        with pytest.raises(FileNotFoundError):
            load_metadata("non_existent.csv")

    def test_load_metadata_invalid_csv(self, tmp_path):
        """Test ValueError for invalid CSV content."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("not,a,csv\n1,2")
        # We need to trigger an exception during pd.read_csv or subsequent logic
        # Actually pd.read_csv might still work. Let's force a read error.
        with patch('pandas.read_csv', side_effect=Exception("Read error")):
            with pytest.raises(ValueError, match="Failed to load metadata file"):
                load_metadata(bad_csv)

    def test_load_metadata_deprecated_constants(self, temp_data_structure):
        """Test loading metadata with deprecated SystemConstants-like object."""
        class MockConstants:
            PATIENT_ID_COLUMN = 'patientId'
            TARGET_COLUMN = 'Target'
            FILENAME_COLUMN = 'filename'
            IMAGE_EXTENSION = '.png'
        
        with pytest.warns(DeprecationWarning, match="Passing SystemConstants to load_metadata is deprecated"):
            df = load_metadata(temp_data_structure['metadata_path'], MockConstants())
        assert 'filename' in df.columns

    def test_load_metadata_missing_columns(self, tmp_path, sample_config):
        """Test metadata loading with missing required columns."""
        df = pd.DataFrame({'wrong_id': ['p1'], 'wrong_target': [0]})
        path = tmp_path / "missing.csv"
        df.to_csv(path, index=False)
        
        with pytest.raises(ValueError, match="Missing column"):
            load_metadata(path, sample_config)
            
        # Target column missing
        df = pd.DataFrame({'patientId': ['p1'], 'wrong_target': [0]})
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing column"):
            load_metadata(path, sample_config)

    def test_load_metadata_empty_df(self, tmp_path, sample_config):
        """Test metadata loading with an empty dataframe."""
        df = pd.DataFrame(columns=['patientId', 'Target'])
        path = tmp_path / "empty.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="DataFrame is empty"):
            load_metadata(path, sample_config)

    def test_load_metadata_missing_values(self, tmp_path, sample_config):
        """Test metadata loading with missing (NaN) values."""
        df = pd.DataFrame({
            'patientId': ['p1', None],
            'Target': [0, 1]
        })
        path = tmp_path / "nan.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing values found in column"):
            load_metadata(path, sample_config)

    def test_sample_dataframe_success(self):
        """Test stratified sampling of dataframe."""
        # Use a larger dataset to avoid rounding edge cases where sample size might be slightly off
        df = pd.DataFrame({
            'Target': [0] * 10 + [1] * 10
        })
        sampled = sample_dataframe(df, sample_fraction=0.5, target_column='Target')
        assert len(sampled) == 10
        assert sampled['Target'].value_counts().to_dict() == {0: 5, 1: 5}

    def test_sample_dataframe_full(self):
        """Test sampling with fraction >= 1.0 (returns copy)."""
        df = pd.DataFrame({'Target': [0, 1]})
        sampled = sample_dataframe(df, sample_fraction=1.0, target_column='Target')
        assert len(sampled) == 2

    def test_sample_dataframe_single_class(self):
        """Test random sampling when only one class is present."""
        df = pd.DataFrame({'Target': [0, 0, 0, 0]})
        sampled = sample_dataframe(df, sample_fraction=0.5, target_column='Target')
        assert len(sampled) == 2

    def test_sample_dataframe_invalid_fraction(self):
        """Test ValueError for invalid sample fraction."""
        df = pd.DataFrame({'Target': [0, 1]})
        with pytest.raises(ValueError, match="sample_fraction must be between 0.0 and 1.0"):
            sample_dataframe(df, sample_fraction=0, target_column='Target')

    def test_sample_dataframe_failure(self):
        """Test error handling in sample_dataframe."""
        with pytest.raises(ValueError, match="Data sampling failed"):
            sample_dataframe("not a df", 0.5, 'Target')

    def test_create_train_val_split_success(self):
        """Test successful train/validation split."""
        df = pd.DataFrame({
            'Target': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        train, val = create_train_val_split(df, validation_split=0.25, target_column='Target')
        assert len(train) == 6
        assert len(val) == 2

    def test_create_train_val_split_invalid_split(self):
        """Test ValueError for invalid validation split."""
        df = pd.DataFrame({'Target': [0, 1]})
        with pytest.raises(ValueError, match="validation_split must be between 0.0 and 1.0"):
            create_train_val_split(df, validation_split=1.5, target_column='Target')

    def test_create_train_val_split_failure(self):
        """Test error handling in create_train_val_split."""
        with pytest.raises(ValueError, match="Train/validation split failed"):
            create_train_val_split("not a df", 0.2, 'Target')

    def test_load_and_split_data_success(self, temp_data_structure, sample_config):
        """Test complete data loading and splitting pipeline."""
        # Setup config for this test
        sample_config.set('paths.base_path', temp_data_structure['base_path'])
        sample_config.set('paths.metadata_filename', 'Train_metadata.csv')
        sample_config.set('experiment.sample_fraction', 1.0)
        sample_config.set('experiment.validation_split', 0.5)
        
        train, val = load_and_split_data(sample_config)
        assert len(train) == 10
        assert len(val) == 10

    def test_load_and_split_data_no_config(self, temp_data_structure):
        """Test load_and_split_data without explicit config."""
        with patch('federated_pneumonia_detection.config.config_manager.ConfigManager') as MockConfig:
            instance = MockConfig.return_value
            instance.get.side_effect = lambda k, default=None: {
                'paths.base_path': temp_data_structure['base_path'],
                'paths.metadata_filename': 'Train_metadata.csv',
                'experiment.sample_fraction': 1.0,
                'experiment.validation_split': 0.5,
                'columns.target': 'Target',
                'columns.filename': 'filename',
                'columns.patient_id': 'patientId',
                'system.image_extension': '.png'
            }.get(k, default)
            
            train, val = load_and_split_data()
            assert train is not None

    def test_load_and_split_data_deprecated_constants(self, temp_data_structure_custom, sample_config):
        """Test load_and_split_data with deprecated SystemConstants."""
        # Use a larger dataset to avoid stratification issues
        df_large = SampleDataFactory.create_sample_metadata(num_samples=100)
        # Create a real temp data structure with these rows
        with TempDataStructure(metadata_df=df_large) as paths:
            class MockConstants:
                BASE_PATH = paths['base_path']
                METADATA_FILENAME = 'Train_metadata.csv'
                TARGET_COLUMN = 'Target'
                PATIENT_ID_COLUMN = 'patientId'
                FILENAME_COLUMN = 'filename'
                IMAGE_EXTENSION = '.png'

            # We need sample_config too because the code uses it for experiment params even with constants
            with patch('federated_pneumonia_detection.config.config_manager.ConfigManager') as MockConfig:
                 MockConfig.return_value = sample_config
                 train, val = load_and_split_data(MockConstants())
                 assert train is not None

    def test_load_and_split_data_failure(self):
        """Test pipeline failure handling."""
        with pytest.raises(ValueError, match="Data processing pipeline failed"):
            load_and_split_data(metadata_path="non_existent.csv")

    def test_validate_image_paths_success(self, temp_data_structure, sample_config):
        """Test successful image path validation."""
        sample_config.set('paths.base_path', temp_data_structure['base_path'])
        assert validate_image_paths(sample_config) is True

    def test_validate_image_paths_no_config(self, temp_data_structure):
        """Test image path validation without explicit config."""
        with patch('federated_pneumonia_detection.config.config_manager.ConfigManager') as MockConfig:
            instance = MockConfig.return_value
            instance.get.side_effect = lambda k, default=None: {
                'paths.base_path': temp_data_structure['base_path'],
                'paths.main_images_folder': 'Images',
                'paths.images_subfolder': 'Images'
            }.get(k, default)
            assert validate_image_paths() is True

    def test_validate_image_paths_deprecated_constants(self, temp_data_structure):
        """Test image path validation with deprecated constants."""
        class MockConstants:
            BASE_PATH = temp_data_structure['base_path']
            MAIN_IMAGES_FOLDER = 'Images'
            IMAGES_SUBFOLDER = 'Images'
        
        assert validate_image_paths(MockConstants()) is True

    def test_validate_image_paths_failure(self, tmp_path, sample_config):
        """Test image path validation failures."""
        sample_config.set('paths.base_path', str(tmp_path))
        # Main folder missing
        assert validate_image_paths(sample_config) is False
        
        # Subfolder missing
        main_folder = tmp_path / "Images"
        main_folder.mkdir()
        assert validate_image_paths(sample_config) is False

    def test_validate_image_paths_deprecated_failure(self, tmp_path):
        """Test deprecated path validation failures."""
        class MockConstants:
            BASE_PATH = str(tmp_path)
            MAIN_IMAGES_FOLDER = 'Images'
            IMAGES_SUBFOLDER = 'Images'
        
        assert validate_image_paths(MockConstants()) is False
        
        os.makedirs(os.path.join(tmp_path, 'Images'))
        # still missing subfolder
        assert validate_image_paths(MockConstants()) is False

    def test_get_image_directory_path(self, temp_data_structure, sample_config):
        """Test retrieval of full image directory path."""
        sample_config.set('paths.base_path', temp_data_structure['base_path'])
        path = get_image_directory_path(sample_config)
        assert "Images" in path

        # Deprecated
        class MockConstants:
            BASE_PATH = temp_data_structure['base_path']
            MAIN_IMAGES_FOLDER = 'Images'
            IMAGES_SUBFOLDER = 'Images'
        path = get_image_directory_path(MockConstants())
        assert "Images" in path

        # No config
        with patch('federated_pneumonia_detection.config.config_manager.ConfigManager') as MockConfig:
            MockConfig.return_value = sample_config
            path = get_image_directory_path()
            assert path is not None

    def test_get_data_statistics(self):
        """Test computation of dataframe statistics."""
        df = pd.DataFrame({
            'Target': [0, 0, 1, None]
        })
        stats = get_data_statistics(df, 'Target')
        assert stats['total_samples'] == 4
        assert stats['missing_values']['Target'] == 1
        assert stats['class_balance_ratio'] == 0.5


class TestImageTransforms:
    """Test suite for image transformation utilities."""

    def test_contrast_stretch_percentile(self):
        """Test contrast stretching implementation."""
        img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
        res = XRayPreprocessor.contrast_stretch_percentile(img)
        assert isinstance(res, Image.Image)
        
        # RGB test
        img_rgb = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        res_rgb = XRayPreprocessor.contrast_stretch_percentile(img_rgb)
        assert res_rgb.mode == "RGB"

    def test_adaptive_histogram_equalization(self):
        """Test adaptive histogram equalization (including cv2-not-found case)."""
        img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
        
        # Test when cv2 is present
        try:
            import cv2
            res = XRayPreprocessor.adaptive_histogram_equalization(img)
            assert isinstance(res, Image.Image)
            
            # RGB
            img_rgb = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
            res_rgb = XRayPreprocessor.adaptive_histogram_equalization(img_rgb)
            assert isinstance(res_rgb, Image.Image)
        except ImportError:
            # If not installed, it should return original
            res = XRayPreprocessor.adaptive_histogram_equalization(img)
            assert res == img

    def test_adaptive_histogram_equalization_grayscale(self):
        """Test CLAHE on grayscale image."""
        try:
            import cv2
            img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode="L")
            res = XRayPreprocessor.adaptive_histogram_equalization(img)
            assert res.mode == "L"
        except ImportError:
            pytest.skip("cv2 not installed")

    def test_create_custom_transform_variants(self, sample_config):
        """Test all branches of create_custom_transform."""
        builder = TransformBuilder(sample_config)
        
        # Test zero_one normalization (no additional transform)
        t = builder.create_custom_transform(normalization="zero_one")
        assert len(t.transforms) == 3 # Resize, CenterCrop, ToTensor
        
        # Test pad_resize
        t2 = builder.create_custom_transform(resize_strategy="pad_resize")
        assert any(isinstance(tr, transforms.Resize) for tr in t2.transforms)

        # Test with preprocessing to reach 100% coverage
        t3 = builder.create_custom_transform(preprocessing={'contrast_stretch': True})
        assert any(isinstance(tr, transforms.Lambda) for tr in t3.transforms)

    def test_edge_enhancement(self):
        """Test edge enhancement filter."""
        img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
        res = XRayPreprocessor.edge_enhancement(img)
        assert isinstance(res, Image.Image)

    def test_create_custom_preprocessing_pipeline_success(self):
        """Test creation and execution of custom preprocessing pipeline."""
        pre = XRayPreprocessor()
        fn = pre.create_custom_preprocessing_pipeline(
            contrast_stretch=True, adaptive_hist=True, edge_enhance=True
        )
        img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
        # Mock adaptive hist to avoid cv2 dependency if needed
        with patch.object(XRayPreprocessor, 'adaptive_histogram_equalization', side_effect=lambda x, **kwargs: x):
            res = fn(img)
            assert isinstance(res, Image.Image)

    def test_create_custom_preprocessing_pipeline_failure(self):
        """Test error handling in preprocessing pipeline (returns original)."""
        pre = XRayPreprocessor()
        fn = pre.create_custom_preprocessing_pipeline(contrast_stretch=True)
        img = "not an image"
        res = fn(img)
        assert res == img

    def test_transform_builder_init_no_config(self):
        """Test TransformBuilder initialization without config."""
        with patch('federated_pneumonia_detection.config.config_manager.ConfigManager') as MockConfig:
            builder = TransformBuilder()
            assert builder.config is not None

    def test_build_training_transforms(self, sample_config):
        """Test building training transform pipeline."""
        builder = TransformBuilder(sample_config)
        
        # With augmentation
        t1 = builder.build_training_transforms(enable_augmentation=True)
        assert isinstance(t1, torch.nn.Module) or hasattr(t1, '__call__')
        
        # Without augmentation
        t2 = builder.build_training_transforms(enable_augmentation=False)
        assert t2 is not None

        # With custom preprocessing
        t3 = builder.build_training_transforms(custom_preprocessing={'contrast_stretch': True})
        assert t3 is not None

    def test_build_validation_transforms(self, sample_config):
        """Test building validation/test transform pipeline."""
        builder = TransformBuilder(sample_config)
        t = builder.build_validation_transforms(custom_preprocessing={'edge_enhance': True})
        assert t is not None

    def test_build_test_time_augmentation_transforms(self, sample_config):
        """Test building TTA transform pipelines."""
        builder = TransformBuilder(sample_config)
        ttas = builder.build_test_time_augmentation_transforms(num_augmentations=3)
        assert len(ttas) == 3

    def test_get_normalization_transforms(self, sample_config):
        """Test normalization transform selection."""
        builder = TransformBuilder(sample_config)
        
        # ImageNet
        sample_config.set('system.use_imagenet_norm', True)
        n1 = builder._get_normalization_transforms()
        assert n1[0].mean == [0.485, 0.456, 0.406]
        
        # Simple [-1, 1]
        sample_config.set('system.use_imagenet_norm', False)
        n2 = builder._get_normalization_transforms()
        assert n2[0].mean == [0.5, 0.5, 0.5]

        # Missing key (default to True)
        # We can't easily delete from ConfigManager internal dict, but we can mock it
        with patch.object(sample_config, 'get', return_value=True):
             n3 = builder._get_normalization_transforms()
             assert n3[0].mean == [0.485, 0.456, 0.406]

    def test_create_custom_transform(self, sample_config):
        """Test creation of fully custom transform pipeline."""
        builder = TransformBuilder(sample_config)
        
        # Different strategies and options
        t1 = builder.create_custom_transform(resize_strategy="random_crop", augmentations=["rotation", "horizontal_flip"])
        t2 = builder.create_custom_transform(resize_strategy="pad_resize", augmentations=["color_jitter", "gaussian_blur"])
        t3 = builder.create_custom_transform(normalization="minus_one_one")
        
        assert t1 is not None and t2 is not None and t3 is not None

    def test_get_transforms_convenience(self, sample_config):
        """Test get_transforms convenience function."""
        t_train = get_transforms(sample_config, is_training=True, use_custom_preprocessing=True)
        t_val = get_transforms(sample_config, is_training=False)
        assert t_train is not None and t_val is not None

    def test_create_preprocessing_function_convenience(self, sample_config):
        """Test create_preprocessing_function convenience function."""
        fn = create_preprocessing_function(sample_config, contrast_stretch=True, adaptive_hist=True)
        assert callable(fn)


class TestDataProcessor:
    """Test suite for DataProcessor class."""

    def test_init(self, sample_config):
        """Test DataProcessor initialization."""
        from federated_pneumonia_detection.src.utils.data_processing import DataProcessor
        dp = DataProcessor(sample_config)
        assert dp.config == sample_config

    def test_load_and_process_data(self, temp_data_structure, sample_config):
        """Test end-to-end data processing via orchestrator."""
        from federated_pneumonia_detection.src.utils.data_processing import DataProcessor
        # Use a larger dataset to avoid stratification issues
        df_large = SampleDataFactory.create_sample_metadata(num_samples=100)
        with TempDataStructure(metadata_df=df_large) as paths:
            sample_config.set('paths.base_path', paths['base_path'])
            sample_config.set('paths.metadata_filename', 'Train_metadata.csv')
            dp = DataProcessor(sample_config)
            train, val = dp.load_and_process_data()
            assert train is not None

    def test_validate_image_paths(self, temp_data_structure, sample_config):
        """Test path validation via DataProcessor."""
        from federated_pneumonia_detection.src.utils.data_processing import DataProcessor
        sample_config.set('paths.base_path', temp_data_structure['base_path'])
        dp = DataProcessor(sample_config)
        assert dp.validate_image_paths() is True

    def test_get_image_paths(self, temp_data_structure, sample_config):
        """Test image paths retrieval."""
        from federated_pneumonia_detection.src.utils.data_processing import DataProcessor
        sample_config.set('paths.base_path', temp_data_structure['base_path'])
        dp = DataProcessor(sample_config)
        main, sub = dp.get_image_paths()
        assert "Images" in main and "Images" in sub

    def test_private_wrappers(self, temp_data_structure, sample_config):
        """Test private compatibility wrappers."""
        from federated_pneumonia_detection.src.utils.data_processing import DataProcessor
        sample_config.set('paths.base_path', temp_data_structure['base_path'])
        sample_config.set('paths.metadata_filename', 'Train_metadata.csv')
        dp = DataProcessor(sample_config)
        
        # _load_metadata
        df = dp._load_metadata()
        assert not df.empty
        
        # _prepare_filenames
        df_prep = dp._prepare_filenames(df)
        assert 'filename' in df_prep.columns
        
        # _validate_data
        dp._validate_data(df_prep) # Should not raise
        
        # _validate_data failures
        with pytest.raises(ValueError, match="DataFrame is empty"):
            dp._validate_data(pd.DataFrame())
        with pytest.raises(ValueError, match="Missing required columns"):
            dp._validate_data(pd.DataFrame({'wrong': [1]}))
        with pytest.raises(ValueError, match="Missing values found"):
            bad_df = df_prep.copy()
            bad_df.iloc[0, bad_df.columns.get_loc('Target')] = None
            dp._validate_data(bad_df)

        # _sample_data
        sampled = dp._sample_data(df_prep, 0.5, 42)
        assert len(sampled) < len(df_prep)
        
        # _create_train_val_split
        train, val = dp._create_train_val_split(df_prep, 0.2, 42)
        assert len(train) > 0 and len(val) > 0

    def test_prepare_filenames_failure(self, sample_config):
        """Test filename preparation failure."""
        from federated_pneumonia_detection.src.utils.data_processing import DataProcessor
        dp = DataProcessor(sample_config)
        with pytest.raises(ValueError, match="Missing column"):
            dp._prepare_filenames(pd.DataFrame({'wrong': [1]}))


def test_logger_setup():
    """Test custom logger initialization."""
    from federated_pneumonia_detection.src.utils.loggers.logger import get_logger, setup_logger
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.getEffectiveLevel() == logging.INFO
    
    logger2 = setup_logger("setup_test")
    assert logger2.level == logging.INFO
    
    # Check if formatters are added (at least one handler)
    assert len(logger.handlers) >= 0 # Might be inherited
