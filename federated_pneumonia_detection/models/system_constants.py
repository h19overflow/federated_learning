"""
System constants and global configuration values.
Centralizes all configurable parameters for the federated pneumonia detection system.
"""

from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConstants:
    """
    Global configuration values for the federated pneumonia detection system.

    All parameters are configurable and have sensible defaults.
    """

    # Image processing constants
    IMG_SIZE: Tuple[int, int] = (224, 224)
    IMAGE_EXTENSION: str = '.png'

    # Training constants
    BATCH_SIZE: int = 128
    SAMPLE_FRACTION: float = 0.05
    VALIDATION_SPLIT: float = 0.20
    SEED: int = 42

    # File paths (relative to project root)
    BASE_PATH: str = '.'
    MAIN_IMAGES_FOLDER: str = 'Images'
    IMAGES_SUBFOLDER: str = 'Images'
    METADATA_FILENAME: str = 'Train_metadata.csv'

    # Data column names
    PATIENT_ID_COLUMN: str = 'patientId'
    TARGET_COLUMN: str = 'Target'
    FILENAME_COLUMN: str = 'filename'


    @classmethod
    def create_custom(
        cls,
        img_size: Tuple[int, ...] = (224, 224),
        batch_size: int = 128,
        sample_fraction: float = 0.10,
        validation_split: float = 0.20,
        seed: int = 42,
        base_path: str = '.',
        main_images_folder: str = 'Images',
        images_subfolder: str = 'Images',
        metadata_filename: str = 'Train_metadata.csv',
        image_extension: str = '.png'
    ) -> 'SystemConstants':
        """
        Create custom system constants with specified values.

        Args:
            img_size: Target image size for preprocessing
            batch_size: Batch size for training
            sample_fraction: Fraction of data to sample for experiments
            validation_split: Fraction of data for validation
            seed: Random seed for reproducibility
            base_path: Base path for data files
            main_images_folder: Main folder containing images
            images_subfolder: Subfolder within main images folder
            metadata_filename: Name of metadata CSV file
            image_extension: File extension for images

        Returns:
            SystemConstants instance with custom values
        """
        return cls(
            IMG_SIZE=img_size,
            BATCH_SIZE=batch_size,
            SAMPLE_FRACTION=sample_fraction,
            VALIDATION_SPLIT=validation_split,
            SEED=seed,
            BASE_PATH=base_path,
            MAIN_IMAGES_FOLDER=main_images_folder,
            IMAGES_SUBFOLDER=images_subfolder,
            METADATA_FILENAME=metadata_filename,
            IMAGE_EXTENSION=image_extension
        )
if __name__ == '__main__':
    constants = SystemConstants()
    print(constants)
    custom_constants = SystemConstants.create_custom(img_size=(256, 256), batch_size=64, seed=123)
    print(custom_constants)