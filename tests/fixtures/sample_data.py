"""
Sample data fixtures for testing.
Provides reusable test data for various test scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import tempfile
from PIL import Image


class SampleDataFactory:
    """Factory class for creating test data fixtures."""

    @staticmethod
    def create_dummy_image(
        size: Tuple[int, int] = (224, 224),
        color_mode: str = 'RGB',
        noise_level: float = 0.5
    ) -> Image.Image:
        """
        Create a dummy image with random noise.

        Args:
            size: Image dimensions (width, height)
            color_mode: 'RGB' or 'L' (grayscale)
            noise_level: Strength of random noise

        Returns:
            PIL Image object
        """
        if color_mode == 'RGB':
            data = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        else:
            data = np.random.randint(0, 255, (size[1], size[0]), dtype=np.uint8)

        return Image.fromarray(data, mode=color_mode)

    @staticmethod
    def create_sample_metadata(
        num_samples: int = 20,
        class_balance: float = 0.5,
        additional_columns: bool = True
    ) -> pd.DataFrame:
        """
        Create sample metadata DataFrame for testing.

        Args:
            num_samples: Number of samples to create
            class_balance: Ratio of class 1 to total samples
            additional_columns: Whether to include additional metadata columns

        Returns:
            Sample metadata DataFrame
        """
        np.random.seed(42)  # For reproducible test data

        # Create patient IDs
        patient_ids = [f"patient_{i:04d}" for i in range(num_samples)]

        # Create balanced targets
        num_positive = int(num_samples * class_balance)
        targets = [1] * num_positive + [0] * (num_samples - num_positive)
        np.random.shuffle(targets)

        # Base columns
        data = {
            'patientId': patient_ids,
            'Target': targets
        }

        # Add additional realistic columns if requested
        if additional_columns:
            data.update({
                'age': np.random.randint(20, 90, num_samples),
                'gender': np.random.choice(['M', 'F'], num_samples),
                'view': np.random.choice(['PA', 'AP'], num_samples),
                'finding': np.random.choice(['Normal', 'Pneumonia'], num_samples)
            })

        return pd.DataFrame(data)

    @staticmethod
    def create_imbalanced_metadata(num_samples: int = 100) -> pd.DataFrame:
        """Create imbalanced dataset for testing edge cases."""
        return SampleDataFactory.create_sample_metadata(
            num_samples=num_samples,
            class_balance=0.1  # 10% positive class
        )

    @staticmethod
    def create_single_class_metadata(num_samples: int = 50) -> pd.DataFrame:
        """Create single-class dataset for testing edge cases."""
        patient_ids = [f"patient_{i:04d}" for i in range(num_samples)]
        return pd.DataFrame({
            'patientId': patient_ids,
            'Target': [0] * num_samples  # All negative class
        })

    @staticmethod
    def create_minimal_metadata() -> pd.DataFrame:
        """Create minimal valid dataset."""
        return pd.DataFrame({
            'patientId': ['patient_001', 'patient_002'],
            'Target': [0, 1]
        })

    @staticmethod
    def create_corrupted_metadata() -> Dict[str, pd.DataFrame]:
        """Create various corrupted datasets for error testing."""
        return {
            'missing_patient_id': pd.DataFrame({
                'wrongColumn': ['001', '002'],
                'Target': [0, 1]
            }),
            'missing_target': pd.DataFrame({
                'patientId': ['001', '002'],
                'wrongTarget': [0, 1]
            }),
            'empty_dataframe': pd.DataFrame(),
            'with_nulls': pd.DataFrame({
                'patientId': ['001', None, '003'],
                'Target': [0, 1, None]
            }),
            'wrong_types': pd.DataFrame({
                'patientId': [1, 2, 3],  # Should be strings
                'Target': ['a', 'b', 'c']  # Should be numeric
            })
        }


class TempDataStructure:
    """Context manager for creating temporary data structures for testing."""

    def __init__(
        self,
        metadata_df: pd.DataFrame = None,
        create_images: bool = True,
        images_format: str = '.png',
        color_mode: str = 'RGB'
    ):
        """
        Initialize temporary data structure.

        Args:
            metadata_df: DataFrame to save as metadata CSV
            create_images: Whether to create dummy image files
            images_format: File extension for images
            color_mode: 'RGB' or 'L' for generated images
        """
        self.metadata_df = metadata_df if metadata_df is not None else SampleDataFactory.create_sample_metadata()
        self.create_images = create_images
        self.images_format = images_format
        self.color_mode = color_mode
        self.temp_dir = None
        self.paths = {}

    def __enter__(self) -> Dict[str, str]:
        """Create temporary directory structure."""
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)

        # Create directory structure
        images_dir = temp_path / "Images" / "Images"
        images_dir.mkdir(parents=True)

        # Create dummy image files if requested
        if self.create_images:
            for patient_id in self.metadata_df['patientId']:
                image_file = images_dir / f"{patient_id}{self.images_format}"
                img = SampleDataFactory.create_dummy_image(color_mode=self.color_mode)
                img.save(image_file)

        # Save metadata CSV
        metadata_path = temp_path / "Train_metadata.csv"
        self.metadata_df.to_csv(metadata_path, index=False)

        # Store paths for easy access
        self.paths = {
            'base_path': str(temp_path),
            'metadata_path': str(metadata_path),
            'images_dir': str(images_dir),
            'main_images_dir': str(temp_path / "Images")
        }

        return self.paths

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory."""
        if self.temp_dir:
            self.temp_dir.cleanup()


class MockDatasets:
    """Mock datasets for testing different scenarios."""

    @staticmethod
    def pneumonia_dataset() -> pd.DataFrame:
        """Create realistic pneumonia dataset with proper distribution."""
        np.random.seed(42)

        # Create more realistic patient distribution
        num_samples = 200
        patient_ids = [f"PAT_{i:05d}" for i in range(1, num_samples + 1)]

        # Age distribution typical for pneumonia studies
        ages = np.random.normal(50, 20, num_samples)
        ages = np.clip(ages, 18, 95).astype(int)

        # Gender distribution
        genders = np.random.choice(['M', 'F'], num_samples, p=[0.52, 0.48])

        # Pneumonia prevalence (roughly 30% positive)
        targets = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])

        # View types
        views = np.random.choice(['PA', 'AP', 'Lateral'], num_samples, p=[0.6, 0.3, 0.1])

        return pd.DataFrame({
            'patientId': patient_ids,
            'Target': targets,
            'age': ages,
            'gender': genders,
            'view': views,
            'institution': np.random.choice(['Hospital_A', 'Hospital_B', 'Hospital_C'], num_samples)
        })

    @staticmethod
    def federated_datasets() -> List[pd.DataFrame]:
        """Create multiple datasets simulating federated learning clients."""
        np.random.seed(42)

        datasets = []
        for client_id in range(5):  # 5 federated clients
            # Each client has different data distribution
            num_samples = np.random.randint(50, 150)
            patient_ids = [f"CLIENT{client_id}_PAT_{i:03d}" for i in range(num_samples)]

            # Different class imbalance per client
            class_probabilities = [
                [0.8, 0.2],  # Client 0: mostly negative
                [0.6, 0.4],  # Client 1: moderate
                [0.5, 0.5],  # Client 2: balanced
                [0.3, 0.7],  # Client 3: mostly positive
                [0.9, 0.1]   # Client 4: very imbalanced
            ]

            targets = np.random.choice([0, 1], num_samples, p=class_probabilities[client_id])

            df = pd.DataFrame({
                'patientId': patient_ids,
                'Target': targets,
                'client_id': client_id
            })

            datasets.append(df)

        return datasets


def create_test_config_dict() -> Dict:
    """Create test configuration dictionary."""
    return {
        'system': {
            'img_size': [224, 224],
            'batch_size': 32,
            'sample_fraction': 0.5,
            'validation_split': 0.2,
            'seed': 42
        },
        'experiment': {
            'learning_rate': 0.001,
            'epochs': 5,
            'num_clients': 3,
            'num_rounds': 2
        },
        'paths': {
            'base_path': 'test_data',
            'metadata_filename': 'test_metadata.csv'
        }
    }


def create_experiment_scenarios() -> Dict[str, Dict]:
    """Create different experiment scenario configurations."""
    return {
        'quick_test': {
            'system': {'sample_fraction': 0.1, 'batch_size': 16},
            'experiment': {'epochs': 1, 'num_rounds': 1}
        },
        'full_centralized': {
            'system': {'sample_fraction': 1.0},
            'experiment': {'epochs': 10, 'learning_rate': 0.001}
        },
        'federated_simulation': {
            'experiment': {
                'num_clients': 5,
                'clients_per_round': 3,
                'num_rounds': 10,
                'local_epochs': 2
            }
        },
        'gpu_optimized': {
            'system': {'batch_size': 128, 'num_workers': 8},
            'experiment': {'device': 'cuda'}
        }
    }