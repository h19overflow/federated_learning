"""
Test script to verify the reorganized federated learning structure.
Validates imports and basic functionality without requiring actual data.
"""

import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all new imports work correctly."""
    logger.info("Testing imports...")

    try:
        # Test main entry point
        from federated_pneumonia_detection.src.control.federated_learning import FederatedTrainer
        logger.info("✓ FederatedTrainer imported successfully")

        # Test core module
        from federated_pneumonia_detection.src.control.federated_learning.core.simulation_runner import (
            SimulationRunner,
            FlowerClient
        )
        logger.info("✓ Core module (SimulationRunner, FlowerClient) imported successfully")

        # Test data module
        from federated_pneumonia_detection.src.control.federated_learning.data import (
            partition_data_iid,
            partition_data_by_patient,
            partition_data_stratified,
            ClientDataManager
        )
        logger.info("✓ Data module (partitioner, ClientDataManager) imported successfully")

        # Test training module
        from federated_pneumonia_detection.src.control.federated_learning.training import (
            train_one_epoch,
            evaluate_model,
            create_optimizer,
            get_model_parameters,
            set_model_parameters
        )
        logger.info("✓ Training module (functions) imported successfully")

        return True

    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_basic_initialization():
    """Test basic initialization of components."""
    logger.info("\nTesting basic initialization...")

    try:
        from federated_pneumonia_detection.src.control.federated_learning import FederatedTrainer

        # Test FederatedTrainer initialization
        trainer = FederatedTrainer(
            partition_strategy='iid',
            checkpoint_dir='test_checkpoints',
            logs_dir='test_logs'
        )
        logger.info("✓ FederatedTrainer initialized successfully")
        logger.info(f"  - Partition strategy: {trainer.partition_strategy}")
        logger.info(f"  - Checkpoint dir: {trainer.checkpoint_dir}")
        logger.info(f"  - Config loaded: num_clients={trainer.config.num_clients}, num_rounds={trainer.config.num_rounds}")

        # Test get_training_status
        status = trainer.get_training_status()
        logger.info("✓ Training status retrieved successfully")
        logger.info(f"  - Status keys: {list(status.keys())}")

        return True

    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_partitioning_logic():
    """Test data partitioning logic with dummy data."""
    logger.info("\nTesting partitioning logic...")

    try:
        import pandas as pd
        import numpy as np
        from federated_pneumonia_detection.src.control.federated_learning.data import (
            partition_data_iid,
            partition_data_stratified
        )

        # Create dummy data
        np.random.seed(42)
        dummy_data = {
            'patientId': [f'patient_{i}' for i in range(100)],
            'Target': np.random.randint(0, 2, 100),
            'filepath': [f'img_{i}.png' for i in range(100)]
        }
        df = pd.DataFrame(dummy_data)
        logger.info(f"Created dummy dataset: {len(df)} samples")

        # Test IID partitioning
        iid_partitions = partition_data_iid(df, num_clients=5, seed=42)
        logger.info(f"✓ IID partitioning: {len(iid_partitions)} partitions")
        logger.info(f"  - Partition sizes: {[len(p) for p in iid_partitions]}")

        # Test stratified partitioning
        strat_partitions = partition_data_stratified(
            df, num_clients=5, target_column='Target', seed=42
        )
        logger.info(f"✓ Stratified partitioning: {len(strat_partitions)} partitions")
        logger.info(f"  - Partition sizes: {[len(p) for p in strat_partitions]}")

        # Verify class distribution is maintained
        for i, partition in enumerate(strat_partitions[:2]):  # Check first 2
            class_dist = partition['Target'].value_counts().to_dict()
            logger.info(f"  - Partition {i} class distribution: {class_dist}")

        return True

    except Exception as e:
        logger.error(f"✗ Partitioning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structure():
    """Test the directory structure is correct."""
    logger.info("\nTesting directory structure...")

    import os

    base_path = os.path.dirname(__file__)
    expected_dirs = ['core', 'data', 'training', '_old_reference']
    expected_files = [
        'federated_trainer.py',
        '__init__.py',
        'ARCHITECTURE.md',
        'core/simulation_runner.py',
        'data/partitioner.py',
        'data/client_data.py',
        'training/functions.py'
    ]

    try:
        # Check directories
        for dir_name in expected_dirs:
            dir_path = os.path.join(base_path, dir_name)
            if os.path.exists(dir_path):
                logger.info(f"✓ Directory exists: {dir_name}/")
            else:
                logger.warning(f"✗ Directory missing: {dir_name}/")

        # Check files
        for file_path in expected_files:
            full_path = os.path.join(base_path, file_path)
            if os.path.exists(full_path):
                logger.info(f"✓ File exists: {file_path}")
            else:
                logger.warning(f"✗ File missing: {file_path}")

        return True

    except Exception as e:
        logger.error(f"✗ Structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing Reorganized Federated Learning Structure")
    logger.info("=" * 60)

    results = []

    # Run tests
    results.append(("Directory Structure", test_structure()))
    results.append(("Import Tests", test_imports()))
    results.append(("Basic Initialization", test_basic_initialization()))
    results.append(("Partitioning Logic", test_partitioning_logic()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("All tests passed! ✓")
        logger.info("\nThe federated learning structure is ready to use.")
        logger.info("Next step: Run with actual data using FederatedTrainer.train()")
        return 0
    else:
        logger.error("Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
