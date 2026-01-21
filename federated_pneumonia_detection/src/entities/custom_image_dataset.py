"""
Custom PyTorch Dataset for X-ray image loading and processing.
Handles image file loading, transformations, and label management with comprehensive error handling.
"""

from typing import Tuple, Optional, Union, Callable, TYPE_CHECKING
from pathlib import Path
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from .cust_image_internals.validation import (
    validate_inputs,
    validate_image_files,
    validate_all_images,
)
from .cust_image_internals.image_ops import load_image
from .cust_image_internals.stats import (
    get_class_distribution,
    get_sample_info,
    get_memory_usage_estimate,
)

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager


class CustomImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading X-ray images with robust error handling.

    Handles loading image files, applying transformations, and managing labels
    with comprehensive validation and graceful error recovery.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Union[str, Path],
        config: Optional['ConfigManager'] = None,
        filename_column: Optional[str] = None,
        target_column: Optional[str] = None,
        transform: Optional[Callable] = None,
        color_mode: str = "RGB",
        validate_images: bool = True,
    ):
        """
        Initialize the dataset with validation and error handling.

        Args:
            dataframe: DataFrame containing filename and target columns
            image_dir: Directory containing image files
            config: ConfigManager for configuration
            filename_column: Column name for filenames (uses config if None)
            target_column: Column name for targets (uses config if None)
            transform: Optional transform pipeline to apply to images
            color_mode: 'RGB' or 'L' for color mode
            validate_images: Whether to validate image files during initialization

        Raises:
            ValueError: If dataframe is invalid or required columns missing
            FileNotFoundError: If image directory doesn't exist
        """
        if config is None:
            from federated_pneumonia_detection.config.config_manager import ConfigManager
            config = ConfigManager()

        self.logger = get_logger(__name__)
        self.config = config
        self.transform = transform
        self.color_mode = color_mode.upper()
        self.image_dir = Path(image_dir)

        # Get column names from config if not provided
        self.filename_column = filename_column or config.get('columns.filename', 'filename')
        self.target_column = target_column or config.get('columns.target', 'Target')

        # Validate inputs
        validate_inputs(
            dataframe,
            self.image_dir,
            self.color_mode,
            self.filename_column,
            self.target_column,
        )

        # Handle empty dataframes gracefully
        if dataframe.empty:
            self.logger.info("Empty dataframe provided to dataset")
            self.filenames = np.array([])
            self.labels = np.array([])
            self.valid_indices = np.array([])
        else:
            self.filenames = dataframe[self.filename_column].values
            self.labels = dataframe[self.target_column].astype(float).values

            # Validate images if requested
            if validate_images:
                self.valid_indices = validate_image_files(self.filenames, self.image_dir)
            else:
                self.valid_indices = np.arange(len(self.filenames))

        self.logger.info(
            f"Dataset initialized with {len(self.valid_indices)} valid samples"
        )

    def __len__(self) -> int:
        """Return the number of valid items in the dataset."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item by index with comprehensive error handling.

        Args:
            idx: Index of the item to get

        Returns:
            Tuple of (image_tensor, label_tensor)

        Raises:
            IndexError: If index is out of bounds
            RuntimeError: If image loading fails
        """
        if idx >= len(self.valid_indices) or idx < 0:
            self.logger.error(
                f"Index {idx} out of bounds for dataset of size {len(self.valid_indices)}"
            )
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self.valid_indices)}"
            )

        # Get the actual index from valid indices
        actual_idx = self.valid_indices[idx]
        filename = self.filenames[actual_idx]
        label = self.labels[actual_idx]

        # Load image with error handling
        try:
            image = load_image(filename, self.image_dir, self.color_mode)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            # Apply transforms if specified
            if self.transform:
                image = self.transform(image)

            return image, label_tensor

        except Exception as e:
            self.logger.error(f"Error loading sample {idx} (file: {filename}): {e}")
            raise RuntimeError(f"Failed to load sample {idx}: {e}") from e

    def get_class_distribution(self) -> dict:
        """Get class distribution of valid samples."""
        return get_class_distribution(self.labels, self.valid_indices)

    def get_sample_info(self, idx: int) -> dict:
        """Get detailed information about a sample."""
        return get_sample_info(idx, self.valid_indices, self.filenames, self.labels, self.image_dir)

    def validate_all_images(self) -> Tuple[int, int, list]:
        """Validate all images in the dataset."""
        return validate_all_images(self.filenames, self.image_dir)

    def get_memory_usage_estimate(self) -> dict:
        """Estimate memory usage of the dataset."""
        return get_memory_usage_estimate(
            self.valid_indices,
            self.filenames,
            self.image_dir,
            self.color_mode,
        )
