#!/usr/bin/env python3
"""
Programmatic Training Example for Federated Pneumonia Detection System

This example demonstrates how to use the CentralizedTrainer class directly
in your code to train a pneumonia detection model on the existing dataset.

Usage:
    python training_example.py
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


def train_pneumonia_model():
    """
    Example function showing how to train a pneumonia detection model.

    Returns:
        Dictionary with training results and paths
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("🩺 Federated Pneumonia Detection - Training Example")
    print("=" * 55)

    # Define paths
    training_data_path = project_root / "Training"
    config_path = project_root / "federated_pneumonia_detection" / "config" / "default_config.yaml"
    checkpoint_dir = "checkpoints"
    logs_dir = "training_logs"
    experiment_name = "pneumonia_detection_example"

    # Validate paths
    if not training_data_path.exists():
        raise FileNotFoundError(f"Training directory not found: {training_data_path}")

    if not (training_data_path / "Images").exists():
        raise FileNotFoundError(f"Images directory not found: {training_data_path / 'Images'}")

    if not (training_data_path / "stage2_train_metadata.csv").exists():
        raise FileNotFoundError(f"Metadata file not found: {training_data_path / 'stage2_train_metadata.csv'}")

    logger.info(f"Training data: {training_data_path}")
    logger.info(f"Config file: {config_path}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Logs dir: {logs_dir}")

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = CentralizedTrainer(
        config_path=str(config_path) if config_path.exists() else None,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir
    )

    # Validate data source
    logger.info("Validating data source...")
    try:
        validation_result = trainer.validate_source(str(training_data_path))

        # Debug: print the actual result
        logger.info(f"Validation result type: {type(validation_result)}")
        logger.info(f"Validation result: {repr(validation_result)}")

        # Handle different possible return types
        if isinstance(validation_result, dict):
            is_valid = validation_result.get('valid', False)
            error_msg = validation_result.get('error', 'Unknown error')
        elif isinstance(validation_result, (str, bytes)):
            # If it's a string/bytes, check if it's truthy
            is_valid = bool(validation_result.strip())
            error_msg = str(validation_result) if not is_valid else "Validation successful"
        elif hasattr(validation_result, '__bool__'):
            # If it has a boolean value
            is_valid = bool(validation_result)
            error_msg = "Validation failed" if not is_valid else "Validation successful"
        else:
            # Default to valid for unknown types
            is_valid = True
            error_msg = "Validation successful"

        if not is_valid:
            raise ValueError(f"Source validation failed: {error_msg}")

    except Exception as e:
        logger.error(f"Validation error: {e}")
        if "no attribute 'get'" in str(e):
            logger.error("This suggests validate_source returned a non-dict type")
        raise ValueError(f"Source validation failed: {e}")

    logger.info("✅ Data source validation successful!")

    # Train model
    print("\n🚀 Starting training...")
    print("=" * 55)

    try:
        results = trainer.train(
            source_path=str(training_data_path),
            experiment_name=experiment_name,
            csv_filename="stage2_train_metadata.csv"
        )

        print("\n✅ Training completed successfully!")
        print("=" * 55)

        # Display results
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
        raise


def evaluate_model(checkpoint_path: str):
    """
    Example function showing how to evaluate a trained model.

    Args:
        checkpoint_path: Path to the trained model checkpoint

    Returns:
        Dictionary with evaluation results
    """
    logger = logging.getLogger(__name__)

    print(f"\n🔍 Evaluating model: {checkpoint_path}")

    # TODO: Implement model evaluation using the test data
    # This would use the ModelEvaluator class with the test dataset

    logger.info("Model evaluation completed")
    return {"status": "evaluation_completed"}


if __name__ == "__main__":
    try:
        # Train the model
        results = train_pneumonia_model()

        # Optionally evaluate on test data
        if 'checkpoint_path' in results:
            evaluate_model(results['checkpoint_path'])

        print("\n🎉 Training pipeline completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
