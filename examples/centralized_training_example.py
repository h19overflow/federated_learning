"""
Example usage of the centralized training system.
Demonstrates how to train a pneumonia detection model from a zip file containing dataset.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer


def main():
    """Demonstrate centralized training workflow."""
    print("=== Pneumonia Detection Centralized Training Example ===\n")

    # Example zip file path - replace with your actual zip file
    zip_path = "path/to/your/pneumonia_dataset.zip"

    # Check if example zip exists
    if not os.path.exists(zip_path):
        print("Example dataset not found. Please provide a zip file with:")
        print("1. A CSV file with columns: 'patientId' and 'Target'")
        print("2. Image files (.png, .jpg, .jpeg) named as {patientId}.{extension}")
        print("\nExpected zip structure:")
        print("pneumonia_dataset.zip")
        print("├── metadata.csv")
        print("└── images/")
        print("    ├── patient001.png")
        print("    ├── patient002.png")
        print("    └── ...")
        return

    try:
        # Initialize centralized trainer
        print("1. Initializing centralized trainer...")
        trainer = CentralizedTrainer(
            checkpoint_dir="experiments/checkpoints",
            logs_dir="experiments/logs"
        )
        print("✓ Trainer initialized\n")

        # Validate zip file before training
        print("2. Validating zip file contents...")
        validation = trainer.validate_zip_contents(zip_path)

        if validation['valid']:
            print(f"✓ Validation passed:")
            print(f"  - CSV files found: {len(validation['csv_files'])}")
            print(f"  - Images found: {validation['image_count']}")
            print(f"  - Total files: {validation['total_files']}")
        else:
            print(f"✗ Validation failed: {validation['error']}")
            return

        # Start training
        print("\n3. Starting training from zip file...")
        results = trainer.train_from_zip(
            zip_path=zip_path,
            experiment_name="pneumonia_detection_demo",
            csv_filename=None  # Auto-detect CSV file
        )

        # Display results
        print("\n=== Training Results ===")
        print(f"✓ Training completed successfully!")
        print(f"Best model path: {results['best_model_path']}")
        print(f"Best model score: {results['best_model_score']:.4f}")
        print(f"Final epoch: {results['current_epoch']}")
        print(f"Total training steps: {results['global_step']}")

        if 'final_metrics' in results:
            print("\nFinal Metrics:")
            for metric, value in results['final_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

        print(f"\nModel checkpoints saved to: {results['checkpoint_dir']}")
        print(f"Training logs saved to: {results['logs_dir']}")

        # Display model information
        model_info = results['model_summary']
        print(f"\nModel Information:")
        print(f"  Total parameters: {model_info.get('total_parameters', 'N/A'):,}")
        print(f"  Trainable parameters: {model_info.get('trainable_parameters', 'N/A'):,}")
        print(f"  Learning rate: {model_info.get('learning_rate', 'N/A')}")

    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
    except ValueError as e:
        print(f"✗ Data validation error: {e}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise


def create_sample_dataset():
    """
    Create a sample dataset zip file for testing.
    This is a utility function to help users get started.
    """
    print("Creating sample dataset structure...")

    # This would create a sample dataset - implementation depends on your needs
    sample_csv_content = """patientId,Target
patient001,0
patient002,1
patient003,0
patient004,1
patient005,0
"""

    print("Sample CSV structure:")
    print(sample_csv_content)
    print("\nTo create a real dataset:")
    print("1. Prepare your CSV with 'patientId' and 'Target' columns")
    print("2. Collect corresponding image files")
    print("3. Name images as {patientId}.png (or .jpg)")
    print("4. Create a zip file with CSV and images")


def advanced_training_example():
    """
    Example showing advanced training features.
    """
    print("=== Advanced Training Example ===\n")

    # Custom configuration path
    config_path = "config/custom_experiment.yaml"

    trainer = CentralizedTrainer(
        config_path=config_path if os.path.exists(config_path) else None,
        checkpoint_dir="experiments/advanced_checkpoints",
        logs_dir="experiments/advanced_logs"
    )

    # Check training status
    status = trainer.get_training_status()
    print("Training Configuration:")
    for key, value in status['config'].items():
        print(f"  {key}: {value}")

    print(f"\nCheckpoint directory: {status['checkpoint_dir']}")
    print(f"Logs directory: {status['logs_dir']}")


if __name__ == "__main__":
    # Basic usage example
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print("\n" + "="*60)
    print("Additional Examples:")
    print("1. Create sample dataset structure")
    create_sample_dataset()

    print("\n2. Advanced configuration")
    advanced_training_example()

    print("\nFor more information, check the documentation in:")
    print("- documentation/Phase1/")
    print("- documentation/guidelines.md")