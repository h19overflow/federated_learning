"""
Quick smoke test to verify the simulation setup works correctly.
This doesn't run actual training, just validates the structure.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.control.federated_learning.core.simulation_runner import SimulationRunner
from federated_pneumonia_detection.src.control.federated_learning.data.partitioner import partition_data_stratified
import pandas as pd
import numpy as np

def test_simulation_setup():
    """Test that simulation components can be instantiated properly."""
    print("="*80)
    print("Federated Learning Simulation - Smoke Test")
    print("="*80)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create minimal config
    print("\n[1/5] Creating configuration...")
    constants = SystemConstants()
    config = ExperimentConfig(
        num_clients=3,
        clients_per_round=2,
        num_rounds=1,
        local_epochs=1,
        batch_size=4,
        num_workers=0,
        pin_memory=False
    )
    print("✓ Configuration created")

    # Create sample data
    print("\n[2/5] Creating sample dataset...")
    np.random.seed(42)
    df = pd.DataFrame({
        'patientId': [f'p{i}' for i in range(30)],
        'Target': np.random.choice([0, 1], 30),
        'filename': [f'p{i}.png' for i in range(30)]
    })
    print(f"✓ Dataset created: {len(df)} samples")

    # Partition data
    print("\n[3/5] Partitioning data...")
    partitions = partition_data_stratified(
        df,
        config.num_clients,
        constants.TARGET_COLUMN,
        config.seed,
        logger
    )
    print(f"✓ Created {len(partitions)} partitions")
    for i, p in enumerate(partitions):
        print(f"  Client {i}: {len(p)} samples")

    # Initialize SimulationRunner
    print("\n[4/5] Initializing SimulationRunner...")
    runner = SimulationRunner(
        constants=constants,
        config=config,
        logger=logger
    )
    print("✓ SimulationRunner initialized")

    # Verify runner properties
    print("\n[5/5] Verifying runner properties...")
    assert runner.constants == constants
    assert runner.config == config
    assert runner.logger == logger
    print("✓ All properties verified")

    print("\n" + "="*80)
    print("✅ Smoke test PASSED - Simulation setup is functional!")
    print("="*80)
    print("\nNote: This test only validates setup, not actual training.")
    print("To run a real simulation, you need:")
    print("  1. Actual chest X-ray images")
    print("  2. flwr[simulation] installed")
    print("  3. ray backend installed")
    print("\nRun the full example with:")
    print("  uv run python -m federated_pneumonia_detection.examples.run_federated_simulation_example")

if __name__ == "__main__":
    try:
        test_simulation_setup()
    except Exception as e:
        print(f"\n❌ Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
