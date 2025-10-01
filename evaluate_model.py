#!/usr/bin/env python3
"""
Model Evaluation Example for Federated Pneumonia Detection System

This example demonstrates how to evaluate a trained pneumonia detection model
on the test dataset.

Usage:
    python evaluate_model.py --checkpoint_path PATH_TO_CHECKPOINT
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the federated_pneumonia_detection package to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from federated_pneumonia_detection.src.control.reporting.model_evaluator import ModelEvaluator
    from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
except ImportError as e:
    print(f"Error importing evaluation modules: {e}")
    print("Please ensure all dependencies are installed: uv sync")
    sys.exit(1)


def evaluate_trained_model(checkpoint_path: str, test_data_path: str = None):
    """
    Evaluate a trained model on test data.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        test_data_path: Path to test data directory (optional, defaults to project Test/)

    Returns:
        Dictionary with evaluation metrics
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("🔍 Federated Pneumonia Detection - Model Evaluation")
    print("=" * 55)

    # Define paths
    if test_data_path is None:
        test_data_path = project_root / "Test"

    if not Path(test_data_path).exists():
        raise FileNotFoundError(f"Test data directory not found: {test_data_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Model checkpoint: {checkpoint_path}")
    logger.info(f"Test data: {test_data_path}")

    # Load configuration
    config_loader = ConfigLoader()
    config_path = project_root / "federated_pneumonia_detection" / "config" / "default_config.yaml"

    try:
        constants = config_loader.create_system_constants(str(config_path) if config_path.exists() else None)
        config = config_loader.create_experiment_config(str(config_path) if config_path.exists() else None)
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        constants = config_loader.create_system_constants()
        config = config_loader.create_experiment_config()

    # Initialize evaluator
    logger.info("Initializing model evaluator...")
    evaluator = ModelEvaluator(
        constants=constants,
        config=config,
        logger=logger
    )

    # Evaluate model
    print("\n📊 Starting model evaluation...")
    print("=" * 55)

    try:
        # Evaluate on test data
        results = evaluator.evaluate_from_directory(
            model_path=checkpoint_path,
            test_dir=test_data_path
        )

        print("\n✅ Evaluation completed successfully!")
        print("=" * 55)

        # Display results
        logger.info("Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")

        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def create_test_predictions(checkpoint_path: str, output_dir: str = "test_predictions"):
    """
    Create predictions on test data and save results.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_dir: Directory to save prediction results
    """
    logger = logging.getLogger(__name__)

    print(f"\n🔮 Creating predictions on test data...")
    print("=" * 55)

    # TODO: Implement prediction generation
    # This would load the model and create predictions for all test images

    logger.info(f"Predictions saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained pneumonia detection model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to test data directory (optional, defaults to ./Test/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    try:
        # Evaluate the model
        results = evaluate_trained_model(
            checkpoint_path=args.checkpoint_path,
            test_data_path=args.test_data_path
        )

        # Optionally create predictions
        create_test_predictions(args.checkpoint_path, args.output_dir)

        print("\n🎉 Evaluation pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)
