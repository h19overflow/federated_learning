"""
Utility functions and helper classes for the federated pneumonia detection system.
Contains data processing, file handling, and other support functions.
"""

from .data_processing import (
    load_metadata,
    sample_dataframe,
    create_train_val_split,
    load_and_split_data,
    validate_image_paths,
    get_image_directory_path,
    get_data_statistics,
    DataProcessor  # Deprecated but kept for compatibility
)
from .image_transforms import (
    TransformBuilder,
    XRayPreprocessor,
    get_transforms,
    create_preprocessing_function
)

__all__ = [
    # Data processing functions
    'load_metadata',
    'sample_dataframe',
    'create_train_val_split',
    'load_and_split_data',
    'validate_image_paths',
    'get_image_directory_path',
    'get_data_statistics',
    'DataProcessor',  # Deprecated

    # Image transforms
    'TransformBuilder',
    'XRayPreprocessor',
    'get_transforms',
    'create_preprocessing_function'
]