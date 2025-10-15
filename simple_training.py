#!/usr/bin/env python3
"""
Simple Training Example for Federated Pneumonia Detection System

This is a simplified example that demonstrates how to use the CentralizedTrainer
with basic configuration to train on the existing dataset.

Usage:
    python simple_training.py
"""

import logging
import sys
from pathlib import Path

# Add the federated_pneumonia_detection package to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer
except ImportError as e:
    print(f"Error importing CentralizedTrainer: {e}")
    print("Please ensure all dependencies are installed: uv sync")
    sys.exit(1)


def simple_training_example():
    """
    Simple training example that works with the existing dataset.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 50)

    # Define paths
    training_data_path = project_root / "Training"
    config_path = project_root / "federated_pneumonia_detection" / "config" / "default_config.yaml"
    if not Path.exists('results'):
        Path.mkdir('results')
    checkpoint_dir = "results/simple_checkpoints"
    logs_dir = "results/simple_logs"
    experiment_name = "resuls/simple_pneumonia_training"
    if not Path.exists(checkpoint_dir):
        Path.mkdir(checkpoint_dir)
    if not Path.exists(logs_dir):
        Path.mkdir(logs_dir)

    logger.info(f"Training data: {training_data_path}")
    logger.info(f"Config: {config_path}")

    # Verify data exists
    if not training_data_path.exists():
        raise FileNotFoundError(f"Training directory not found: {training_data_path}")

    images_dir = training_data_path / "Images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    csv_file = training_data_path / "stage2_train_metadata.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    print(f"✅ Found training data: {len(list(images_dir.glob('*.png')))} images")
    print(f"✅ Found metadata: {csv_file}")

    # Initialize trainer with minimal configuration
    print("\n🔧 Initializing trainer...")
    trainer = CentralizedTrainer(
        config_path=str(config_path) if config_path.exists() else None,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir
    )

    # Train model (skip validation that's causing issues)
    print("\n🚀 Starting training...")
    print("=" * 50)

    try:
        # Use the train method directly with the directory path
        results = trainer.train(
            source_path=str(training_data_path),
            experiment_name=experiment_name,
            csv_filename="stage2_train_metadata.csv"
        )

        print("\n✅ Training completed successfully!")
        print("=" * 50)

        # Show results
        logger.info("Training Results:")
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")

        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Print the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        results = simple_training_example()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
