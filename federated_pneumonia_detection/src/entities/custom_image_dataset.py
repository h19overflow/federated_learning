"""
Custom PyTorch Dataset for X-ray image loading and processing.
Handles image file loading, transformations, and label management with comprehensive error handling.
"""

from typing import Tuple, Optional, Union, Callable
from pathlib import Path
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np

from federated_pneumonia_detection.models.system_constants import SystemConstants


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
        constants: SystemConstants,
        transform: Optional[Callable] = None,
        color_mode: str = "RGB",
        validate_images: bool = True,
    ):
        """
        Initialize the dataset with validation and error handling.

        Args:
            dataframe: DataFrame containing 'filename' and 'Target' columns
            image_dir: Directory containing image files
            constants: SystemConstants for configuration
            transform: Optional transform pipeline to apply to images
            color_mode: 'RGB' or 'L' for color mode
            validate_images: Whether to validate image files during initialization

        Raises:
            ValueError: If dataframe is invalid or required columns missing
            FileNotFoundError: If image directory doesn't exist
        """
        self.logger = get_logger(__name__)
        self.constants = constants
        self.transform = transform
        self.color_mode = color_mode.upper()
        self.image_dir = Path(image_dir)

        # Validate inputs
        self._validate_inputs(dataframe, image_dir)

        # Handle empty dataframes gracefully
        if dataframe.empty:
            self.logger.info("Empty dataframe provided to dataset")
            self.filenames = np.array([])
            self.labels = np.array([])
            self.valid_indices = np.array([])
        else:
            self.filenames = dataframe[constants.FILENAME_COLUMN].values
            self.labels = dataframe[constants.TARGET_COLUMN].astype(float).values

            # Validate images if requested
            if validate_images:
                self.valid_indices = self._validate_image_files()
            else:
                self.valid_indices = np.arange(len(self.filenames))

        self.logger.info(
            f"Dataset initialized with {len(self.valid_indices)} valid samples"
        )

    def _validate_inputs(
        self, dataframe: pd.DataFrame, image_dir: Union[str, Path]
    ) -> None:
        """Validate constructor inputs."""
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame")

        if not self.image_dir.exists():
            self.logger.error(f"Image directory not found: {image_dir}")
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if not self.image_dir.is_dir():
            self.logger.error(f"Image directory path is not a directory: {image_dir}")
            raise ValueError(f"Image directory path is not a directory: {image_dir}")

        if self.color_mode not in ["RGB", "L"]:
            self.logger.error(f"Color mode must be 'RGB' or 'L'")
            raise ValueError("color_mode must be 'RGB' or 'L'")

        # Check required columns if dataframe is not empty
        if not dataframe.empty:
            required_columns = [
                self.constants.FILENAME_COLUMN,
                self.constants.TARGET_COLUMN,
            ]
            missing_columns = [
                col for col in required_columns if col not in dataframe.columns
            ]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")

    def _validate_image_files(self) -> np.ndarray:
        """
        Validate that image files exist and are readable.

        Returns:
            Array of valid indices
        """
        valid_indices = []
        invalid_count = 0

        for idx, filename in enumerate(self.filenames):
            image_path = self.image_dir / filename

            try:
                if not image_path.exists():
                    self.logger.info(f"Image file not found: {image_path}")
                    invalid_count += 1
                    continue

                # Try to open the image to validate format
                with Image.open(image_path) as img:
                    # Basic validation - ensure it's a valid image
                    img.verify()

                valid_indices.append(idx)

            except Exception as e:
                self.logger.info(f"Invalid image file {image_path}: {e}")
                invalid_count += 1

        if invalid_count > 0:
            self.logger.info(
                f"Found {invalid_count} invalid image files out of {len(self.filenames)}"
            )

        return np.array(valid_indices)

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
            image = self._load_image(filename)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            # Apply transforms if specified
            if self.transform:
                image = self.transform(image)

            return image, label_tensor

        except Exception as e:
            self.logger.error(f"Error loading sample {idx} (file: {filename}): {e}")
            raise RuntimeError(f"Failed to load sample {idx}: {e}") from e

    def _load_image(self, filename: str) -> Image.Image:
        """
        Load and convert image with error handling.

        Args:
            filename: Name of image file to load

        Returns:
            PIL Image in specified color mode

        Raises:
            RuntimeError: If image loading fails
        """
        image_path = self.image_dir / filename

        try:
            with Image.open(image_path) as img:
                # Convert to RGB first to handle various input formats
                img_rgb = img.convert("RGB")

                # Then convert to target mode if different
                if self.color_mode == "L":
                    return img_rgb.convert("L")
                else:
                    return img_rgb

        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise RuntimeError(f"Failed to load image {image_path}: {e}") from e

    def get_class_distribution(self) -> dict:
        """
        Get class distribution of valid samples.

        Returns:
            Dictionary with class counts
        """
        if len(self.valid_indices) == 0:
            return {}

        valid_labels = self.labels[self.valid_indices]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        return dict(zip(unique_labels.astype(int), counts))

    def get_sample_info(self, idx: int) -> dict:
        """
        Get detailed information about a sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample information
        """
        if idx >= len(self.valid_indices) or idx < 0:
            self.logger.error(f"Index {idx} out of bounds")
            raise IndexError(f"Index {idx} out of bounds")

        actual_idx = self.valid_indices[idx]
        filename = self.filenames[actual_idx]
        label = self.labels[actual_idx]
        image_path = self.image_dir / filename

        info = {
            "index": idx,
            "actual_index": actual_idx,
            "filename": filename,
            "label": label,
            "image_path": str(image_path),
            "exists": image_path.exists(),
        }

        # Add image info if file exists
        if image_path.exists():
            try:
                with Image.open(image_path) as img:
                    info.update(
                        {
                            "image_size": img.size,
                            "image_mode": img.mode,
                            "image_format": img.format,
                        }
                    )
            except Exception as e:
                info["image_error"] = str(e)

        return info

    def validate_all_images(self) -> Tuple[int, int, list]:
        """
        Validate all images in the dataset.

        Returns:
            Tuple of (valid_count, invalid_count, invalid_files)
        """
        invalid_files = []
        valid_count = 0

        for idx in range(len(self.filenames)):
            filename = self.filenames[idx]
            image_path = self.image_dir / filename

            try:
                if not image_path.exists():
                    invalid_files.append((filename, "File not found"))
                    continue

                with Image.open(image_path) as img:
                    img.verify()

                valid_count += 1

            except Exception as e:
                invalid_files.append((filename, str(e)))
                self.logger.error(f"Error validating image {image_path}: {e}")
        invalid_count = len(invalid_files)
        return valid_count, invalid_count, invalid_files

    def get_memory_usage_estimate(self) -> dict:
        """
        Estimate memory usage of the dataset.

        Returns:
            Dictionary with memory usage estimates
        """
        if len(self.valid_indices) == 0:
            self.logger.info("No valid indices in dataset")
            return {"total_samples": 0, "estimated_memory_mb": 0}

        # Sample a few images to estimate average size
        sample_size = min(10, len(self.valid_indices))
        total_pixels = 0

        for i in range(sample_size):
            try:
                actual_idx = self.valid_indices[i]
                filename = self.filenames[actual_idx]
                image_path = self.image_dir / filename

                with Image.open(image_path) as img:
                    total_pixels += img.size[0] * img.size[1]

            except Exception:
                self.logger.error(f"Error validating image {image_path}")
                continue

        if sample_size > 0:
            avg_pixels = total_pixels / sample_size
            channels = 3 if self.color_mode == "RGB" else 1
            bytes_per_pixel = 4  # Assuming float32

            estimated_mb_per_image = (avg_pixels * channels * bytes_per_pixel) / (
                1024 * 1024
            )
            total_estimated_mb = estimated_mb_per_image * len(self.valid_indices)
        else:
            self.logger.info("No valid indices in dataset")
            estimated_mb_per_image = 0
            total_estimated_mb = 0

        return {
            "total_samples": len(self.valid_indices),
            "avg_pixels_per_image": avg_pixels if sample_size > 0 else 0,
            "estimated_mb_per_image": estimated_mb_per_image,
            "estimated_total_memory_mb": total_estimated_mb,
            "color_mode": self.color_mode,
        }
