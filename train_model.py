#!/usr/bin/env python3
"""
Training script for Federated Pneumonia Detection System

This script uses the CentralizedTrainer to train a pneumonia detection model
on the existing dataset in the Training/ directory.

Usage:
    python train_model.py [options]

Options:
    --config_path PATH      Path to configuration file (optional)
    --experiment_name NAME  Name for the training experiment (default: pneumonia_detection)
    --checkpoint_dir DIR    Directory for model checkpoints (default: checkpoints)
    --logs_dir DIR         Directory for training logs (default: training_logs)
    --csv_filename NAME    Specific CSV filename to use (default: stage2_train_metadata.csv)
"""

import argparse
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
    print("Please ensure the federated_pneumonia_detection package is properly installed.")
    print("Try: pip install -e .")
    sys.exit(1)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to configuration file (optional)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="pneumonia_detection",
        help="Name for the training experiment"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="training_logs",
        help="Directory for training logs"
    )
    parser.add_argument(
        "--csv_filename",
        type=str,
        default="stage2_train_metadata.csv",
        help="Specific CSV filename to use"
    )

    # Define config path
    config_path = project_root / "federated_pneumonia_detection" / "config" / "default_config.yaml"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("Federated Pneumonia Detection - Training Script")
    print("=" * 60)

    # Initialize trainer
    logger.info("Initializing CentralizedTrainer...")
    try:
        trainer = CentralizedTrainer(
            config_path=str(config_path) if config_path.exists() else None,
            checkpoint_dir=args.checkpoint_dir,
            logs_dir=args.logs_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        sys.exit(1)

    # Define data source path
    training_data_path = project_root / "Training"
    csv_path = training_data_path / args.csv_filename

    # Validate data source
    logger.info("Validating data source...")
    if not training_data_path.exists():
        logger.error(f"Training directory not found: {training_data_path}")
        sys.exit(1)

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    # Check for Images directory
    images_dir = training_data_path / "Images"
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)

    logger.info(f"Training data path: {training_data_path}")
    logger.info(f"CSV file: {csv_path}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Experiment name: {args.experiment_name}")

    # Validate source
    validation_result = trainer.validate_source(str(training_data_path))
    if not validation_result.get('valid', False):
        logger.error(f"Source validation failed: {validation_result.get('error', 'Unknown error')}")
        sys.exit(1)

    logger.info("Source validation successful!")

    # Start training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    try:
        results = trainer.train(
            source_path=str(training_data_path),
            experiment_name=args.experiment_name,
            csv_filename=args.csv_filename
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)

        # Display results
        logger.info("Training Results:")
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")

        # Show checkpoint location
        if 'checkpoint_path' in results:
            logger.info(f"Model saved to: {results['checkpoint_path']}")

        return results

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
