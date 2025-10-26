"""
Script to run centralized training on the pneumonia dataset.
Uses the Training folder containing images and metadata CSV.
"""

import sys
import logging
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer


def main():
    """Run centralized training on the Training dataset."""
    # add try and except block to catch errors and log them

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Define paths
    # convert the id into int , and capture current timestamp
    run_id = int(time.time())
    source_path = "Training"  # Directory containing Images/ and CSV files
    config_path = None  # Use default configuration
    # Output directories
    checkpoint_dir = "results/centralized/checkpoints"
    logs_dir = "results/centralized/logs"

    logger.info(f"Source path: {source_path}")

    # Create trainer
    trainer = CentralizedTrainer(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
    )

    # Display training status
    status = trainer.get_training_status()

    # Run training
    try:
        logger.info("\nStarting training...")

        results = trainer.train(
            source_path=source_path,
            csv_filename="stage2_train_metadata.csv",
            run_id=run_id
        )

        logger.info(f"  Best model: {results.get('best_checkpoint_path', 'N/A')}")

        if 'final_metrics' in results:
            logger.info("\nFinal Metrics:")
            for key, value in results['final_metrics'].items():
                logger.info(f"  {key}: {value}")

        return 0

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("TRAINING FAILED!")
        logger.error("=" * 80)
        logger.error(f"Error: {type(e).__name__}: {str(e)}")

        import traceback
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())

        return 1


if __name__ == "__main__":
    sys.exit(main())
