"""
Example script demonstrating how to run federated learning simulation.
This shows the complete workflow: data preparation, partitioning, and FL simulation.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.control.federated_learning.data.partitioner import (
    partition_data_stratified,
)
from federated_pneumonia_detection.src.control.federated_learning.core.simulation_runner import SimulationRunner


def setup_logging() -> logging.Logger:
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_sample_dataset(num_samples: int = 200) -> pd.DataFrame:
    """
    Create a sample dataset for demonstration.
    In production, you would load your actual pneumonia dataset.

    Args:
        num_samples: Number of samples to create

    Returns:
        DataFrame with patient data
    """
    np.random.seed(42)

    # Create synthetic patient data
    df = pd.DataFrame({
        'patientId': [f'patient_{i:04d}' for i in range(num_samples)],
        'Target': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),  # 0=Normal, 1=Pneumonia
        'filename': [f'patient_{i:04d}.png' for i in range(num_samples)]
    })

    return df


def run_simulation_example():
    """
    Complete example of running a federated learning simulation.
    """
    logger = setup_logging()
    logger.info("="*80)
    logger.info("Federated Learning Simulation Example")
    logger.info("="*80)

    # ============================================================================
    # Step 1: Create or load dataset
    # ============================================================================
    logger.info("\n[Step 1] Creating sample dataset...")

    # In production, replace this with your actual data loading:
    # df = pd.read_csv("path/to/your/metadata.csv")
    df = create_sample_dataset(num_samples=200)

    logger.info(f"Dataset created: {len(df)} samples")
    logger.info(f"Class distribution:\n{df['Target'].value_counts()}")

    # ============================================================================
    # Step 2: Initialize configuration
    # ============================================================================
    logger.info("\n[Step 2] Initializing configuration...")

    # Create system constants
    constants = SystemConstants()

    # Create experiment configuration
    config = ExperimentConfig(
        # Federated learning parameters
        num_clients=5,              # Total number of federated clients
        clients_per_round=3,        # Clients participating per round
        num_rounds=3,               # Number of federated rounds
        local_epochs=2,             # Local training epochs per client

        # Model parameters
        num_classes=1,              # Binary classification
        dropout_rate=0.3,
        fine_tune_layers_count=10,

        # Training parameters
        batch_size=16,
        learning_rate=0.001,
        weight_decay=0.0001,

        # Data parameters
        validation_split=0.2,
        augmentation_strength=1.0,
        color_mode="rgb",

        # Other parameters
        seed=42,
        num_workers=0,              # Set to 0 for Windows compatibility
        pin_memory=False,

        # Checkpoint directory
        checkpoint_dir="federated_checkpoints"
    )

    logger.info(f"Configuration: {config.num_clients} clients, {config.num_rounds} rounds")

    # ============================================================================
    # Step 3: Partition data across clients
    # ============================================================================
    logger.info("\n[Step 3] Partitioning data across clients...")

    # Choose partitioning strategy:
    # - IID: Random distribution (good for baseline)
    # - Stratified: Maintains class balance (recommended for imbalanced data)
    # - By patient: Each client has different patients (realistic for medical data)

    # Using stratified partitioning for this example
    client_partitions = partition_data_stratified(
        df=df,
        num_clients=config.num_clients,
        target_column=constants.TARGET_COLUMN,
        seed=config.seed,
        logger=logger
    )

    # Alternative options:
    # client_partitions = partition_data_iid(df, config.num_clients, config.seed, logger)
    # client_partitions = partition_data_by_patient(df, config.num_clients, constants.PATIENT_ID_COLUMN, config.seed, logger)

    logger.info(f"Created {len(client_partitions)} partitions")
    for i, partition in enumerate(client_partitions):
        logger.info(f"  Client {i}: {len(partition)} samples")

    # ============================================================================
    # Step 4: Set up image directory
    # ============================================================================
    logger.info("\n[Step 4] Setting up image directory...")

    # In production, use your actual image directory:
    # image_dir = "/path/to/chest_xray_images"

    # For this example, we'll create a mock directory
    # NOTE: This won't work with real training - you need actual images!
    image_dir = str(Path(__file__).parent.parent / "data" / "sample_images")
    logger.info(f"Image directory: {image_dir}")
    logger.warning("⚠️  For real training, ensure images exist at this location!")

    # ============================================================================
    # Step 5: Initialize and run simulation
    # ============================================================================
    logger.info("\n[Step 5] Running federated learning simulation...")

    # Create simulation runner
    runner = SimulationRunner(
        constants=constants,
        config=config,
        logger=logger
    )

    # Run the simulation
    try:
        results = runner.run_simulation(
            client_partitions=client_partitions,
            image_dir=image_dir,
            experiment_name="pneumonia_fl_demo"
        )

        # ========================================================================
        # Step 6: Display results
        # ========================================================================
        logger.info("\n[Step 6] Simulation Results")
        logger.info("="*80)
        logger.info(f"Experiment: {results['experiment_name']}")
        logger.info(f"Status: {results['status']}")
        logger.info(f"Clients: {results['num_clients']}")
        logger.info(f"Rounds: {results['num_rounds']}")

        # Display training metrics
        if results['metrics']['losses_distributed']:
            logger.info("\nDistributed Losses (per round):")
            for round_num, (_, loss) in enumerate(results['metrics']['losses_distributed'], 1):
                logger.info(f"  Round {round_num}: {loss:.4f}")

        if results['metrics']['metrics_distributed']:
            logger.info("\nDistributed Metrics:")
            for round_num, metrics in enumerate(results['metrics']['metrics_distributed'], 1):
                logger.info(f"  Round {round_num}: {metrics}")

        logger.info("\n" + "="*80)
        logger.info("✅ Simulation completed successfully!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"\n❌ Simulation failed: {e}")
        raise


if __name__ == "__main__":
    run_simulation_example()
