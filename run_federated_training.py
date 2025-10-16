"""
Script to run federated learning training on the pneumonia dataset.
Uses the Training folder containing images and metadata CSV.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from federated_pneumonia_detection.src.control.federated_learning.federated_trainer import FederatedTrainer


def main():
    """Run federated learning training on the Training dataset."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("FEDERATED LEARNING TRAINING - Pneumonia Detection")
    logger.info("="*80)

    # Define paths
    source_path = "Training"  # Directory containing Images/ and CSV files
    config_path = None  # Use default configuration
    experiment_name = "pneumonia_federated"
    partition_strategy = "stratified"  # Options: 'iid', 'stratified', 'by_patient'

    # Output directories
    checkpoint_dir = "results/federated/checkpoints"
    logs_dir = "results/federated/logs"

    logger.info(f"Source path: {source_path}")
    logger.info(f"Partition strategy: {partition_strategy}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Logs directory: {logs_dir}")

    # Create trainer
    logger.info("\nInitializing FederatedTrainer...")
    trainer = FederatedTrainer(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        partition_strategy=partition_strategy
    )

    # Display training status
    status = trainer.get_training_status()
    logger.info("\nTrainer Configuration:")
    logger.info(f"  Num Rounds: {status['config']['num_rounds']}")
    logger.info(f"  Num Clients: {status['config']['num_clients']}")
    logger.info(f"  Clients per Round: {status['config']['clients_per_round']}")
    logger.info(f"  Local Epochs: {status['config']['local_epochs']}")
    logger.info(f"  Learning Rate: {status['config']['learning_rate']}")
    logger.info(f"  Batch Size: {status['config']['batch_size']}")
    logger.info(f"  Partition Strategy: {status['partition_strategy']}")

    # Run training
    try:
        logger.info("\nStarting federated learning simulation...")
        logger.info("-"*80)

        results = trainer.train(
            source_path=source_path,
            experiment_name=experiment_name,
            csv_filename="stage2_train_metadata.csv"
        )

        logger.info("\n" + "="*80)
        logger.info("FEDERATED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("\nResults Summary:")
        logger.info(f"  Experiment: {results.get('experiment_name', 'N/A')}")
        logger.info(f"  Status: {results.get('status', 'completed')}")
        logger.info(f"  Num Clients: {results.get('num_clients', 'N/A')}")
        logger.info(f"  Num Rounds: {results.get('num_rounds', 'N/A')}")
        logger.info(f"  Partition Strategy: {results.get('partition_strategy', 'N/A')}")
        logger.info(f"  Checkpoint directory: {checkpoint_dir}")
        logger.info(f"  Logs directory: {logs_dir}")

        # Display metrics if available
        if 'metrics' in results:
            metrics = results['metrics']

            if metrics.get('losses_distributed'):
                logger.info("\nTraining Losses per Round:")
                for round_num, (_, loss) in enumerate(metrics['losses_distributed'], 1):
                    logger.info(f"  Round {round_num}: {loss:.4f}")

            if metrics.get('metrics_distributed'):
                logger.info("\nFinal Round Metrics:")
                final_metrics = metrics['metrics_distributed'][-1] if metrics['metrics_distributed'] else {}
                for key, value in final_metrics.items():
                    logger.info(f"  {key}: {value}")

        return 0

    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("FEDERATED TRAINING FAILED!")
        logger.error("="*80)
        logger.error(f"Error: {type(e).__name__}: {str(e)}")

        import traceback
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())

        return 1


if __name__ == "__main__":
    sys.exit(main())
