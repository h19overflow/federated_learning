"""
Script to run federated learning training on the pneumonia dataset.
Uses the Training folder containing images and metadata CSV.
"""

import sys
import logging
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.control.federated_learning import (
    FederatedTrainer,
)
from federated_pneumonia_detection.src.utils.data_processing import load_metadata, sample_dataframe
from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader

# TODO , Figure out how to extract logs from the trainer in order to show in the terminal on the frontend.
# TODO , Saving the resutls to the database is still pending we need to add a step to save the results to the database.
def main():
    """Run federated learning training on the Training dataset."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("FEDERATED LEARNING TRAINING - Pneumonia Detection")
    logger.info("=" * 80)

    try:
        # Define paths and configuration
        source_path = Path("Training")
        image_dir = source_path / "Images"
        csv_filename = "stage2_train_metadata.csv"
        metadata_path = source_path / csv_filename

        logger.info(f"\nData paths:")
        logger.info(f"  Source: {source_path}")
        logger.info(f"  Images: {image_dir}")
        logger.info(f"  Metadata: {metadata_path}")

        # Validate paths
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load configuration from YAML
        logger.info("\nLoading configuration...")
        config_loader = ConfigLoader()
        yaml_config = config_loader.load_config(
            "federated_pneumonia_detection/config/default_config.yaml"
        )
        constants = config_loader.create_system_constants(yaml_config)
        config = config_loader.create_experiment_config(yaml_config)
        
        logger.info(f"  Num clients: {config.num_clients}")
        logger.info(f"  Num rounds: {config.num_rounds}")
        logger.info(f"  Local epochs: {config.local_epochs}")
        logger.info(f"  Sample fraction: {config.sample_fraction}")

        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("  Device: GPU (CUDA)")
        else:
            device = torch.device("cpu")
            logger.info("  Device: CPU")

        # Load metadata
        logger.info("\nLoading dataset metadata...")
        data_df = load_metadata(metadata_path, constants, logger)
        logger.info(f"  Total samples loaded: {len(data_df)}")
        
        # Apply sampling fraction from config
        logger.info(f"\nApplying sample fraction: {config.sample_fraction}")
        data_df = sample_dataframe(
            data_df,
            sample_fraction=config.sample_fraction,
            target_column=constants.TARGET_COLUMN,
            seed=config.seed,
            logger=logger
        )
        logger.info(f"  Samples after sampling: {len(data_df)}")
        class_dist = data_df[constants.TARGET_COLUMN].value_counts().to_dict()
        logger.info(f"  Class distribution: {class_dist}")

        # Initialize trainer
        logger.info("\nInitializing FederatedTrainer...")
        trainer = FederatedTrainer(
            config=config, constants=constants, device=device
        )

        # Run training
        logger.info("\nStarting federated learning simulation...")
        logger.info("-" * 80)

        results = trainer.train(
            data_df=data_df,
            image_dir=image_dir,
            experiment_name="pneumonia_federated",
        )

        # Display summary
        logger.info("\n" + "=" * 80)
        logger.info("FEDERATED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\nResults Summary:")
        logger.info(f"  Experiment: {results.get('experiment_name')}")
        logger.info(f"  Status: {results.get('status')}")
        logger.info(f"  Num Clients: {results.get('num_clients')}")
        logger.info(f"  Num Rounds: {results.get('num_rounds')}")

        return 0

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("FEDERATED TRAINING FAILED!")
        logger.error("=" * 80)
        logger.error(f"Error: {type(e).__name__}: {str(e)}")

        import traceback

        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())

        return 1


if __name__ == "__main__":
    sys.exit(main())
