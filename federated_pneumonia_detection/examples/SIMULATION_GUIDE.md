# Federated Learning Simulation Guide

## Overview

This guide explains how to use the updated `SimulationRunner` for federated learning with the Flower framework. The code has been updated to use **Flower's modern Message API** (v1.13+).

---

## What Changed?

### API Choice: `start_simulation` (Returns History)

The `SimulationRunner` uses Flower's **`start_simulation`** function because it **returns a `History` object** with training metrics, losses, and evaluation results.

**Why `start_simulation` instead of `run_simulation`?**
- `run_simulation` (v1.13+) returns `None` - no way to access metrics programmatically
- `start_simulation` returns `History` object with all metrics and losses
- `History` object contains: `losses_distributed`, `metrics_distributed`, `losses_centralized`, `metrics_centralized`

**Current Implementation:**
```python
history = start_simulation(
    client_fn=client_fn,              # Function that creates Client instances
    num_clients=10,                   # Number of federated clients
    client_resources={"num_cpus": 1}, # Resource allocation per client
    config=ServerConfig(num_rounds=5),# Server configuration
    strategy=FedAvg(...)              # Federated averaging strategy
)

# Access results
final_loss = history.losses_distributed[-1][1]
metrics = history.metrics_distributed
```

### Key Implementation Details

1. **Client Function**: Returns `Client` (using `.to_client()` on `NumPyClient`)
2. **Context Object**: Client function receives `Context` with `node_id` for client identification
3. **Strategy**: Direct `FedAvg` instance with initial parameters
4. **History Return**: Full training history accessible after simulation completes

---

## Quick Start

### 1. Install Dependencies

Ensure you have Flower >= 1.5.0:

```bash
pip install "flwr[simulation]>=1.5.0"
```

### 2. Run the Example

```bash
# Full example with detailed logging
python -m federated_pneumonia_detection.examples.run_federated_simulation_example --mode full

# Minimal example
python -m federated_pneumonia_detection.examples.run_federated_simulation_example --mode minimal
```

---

## Usage Example

### Complete Workflow

```python
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.control.federated_learning.data.partitioner import partition_data_iid
from federated_pneumonia_detection.src.control.federated_learning.core.simulation_runner import SimulationRunner
import pandas as pd
import logging

# 1. Setup
logger = logging.getLogger(__name__)
constants = SystemConstants()
config = ExperimentConfig(
    num_clients=5,
    clients_per_round=3,
    num_rounds=10,
    local_epochs=2
)

# 2. Load and partition data
df = pd.read_csv("metadata.csv")
partitions = partition_data_iid(df, config.num_clients, config.seed, logger)

# 3. Run simulation
runner = SimulationRunner(constants, config, logger)
results = runner.run_simulation(
    client_partitions=partitions,
    image_dir="/path/to/images",
    experiment_name="my_experiment"
)

# 4. Check results
print(f"Status: {results['status']}")
print(f"Final loss: {results['metrics']['losses_distributed'][-1]}")
```

---

## Data Partitioning Strategies

### 1. IID (Independent and Identically Distributed)

Random distribution across clients - good for baseline experiments:

```python
from federated_pneumonia_detection.src.control.federated_learning.data.partitioner import partition_data_iid

partitions = partition_data_iid(
    df=dataframe,
    num_clients=5,
    seed=42,
    logger=logger
)
```

### 2. Stratified

Maintains class balance across clients - recommended for imbalanced datasets:

```python
from federated_pneumonia_detection.src.control.federated_learning.data.partitioner import partition_data_stratified

partitions = partition_data_stratified(
    df=dataframe,
    num_clients=5,
    target_column="Target",
    seed=42,
    logger=logger
)
```

### 3. By Patient

Each client gets data from different patients - most realistic for medical data:

```python
from federated_pneumonia_detection.src.control.federated_learning.data.partitioner import partition_data_by_patient

partitions = partition_data_by_patient(
    df=dataframe,
    num_clients=5,
    patient_column="patientId",
    seed=42,
    logger=logger
)
```

---

## Configuration Options

### ExperimentConfig Parameters

```python
config = ExperimentConfig(
    # Federated learning
    num_clients=10,              # Total number of clients
    clients_per_round=5,         # Clients per training round
    num_rounds=20,               # Total federated rounds
    local_epochs=5,              # Epochs each client trains locally

    # Model architecture
    num_classes=1,               # 1 for binary, >1 for multiclass
    dropout_rate=0.3,
    fine_tune_layers_count=10,   # Layers to fine-tune in ResNet

    # Training
    batch_size=32,
    learning_rate=0.001,
    weight_decay=0.0001,

    # Data
    validation_split=0.2,
    augmentation_strength="medium",  # "light", "medium", "heavy"
    color_mode="rgb",                # "rgb" or "grayscale"

    # System
    seed=42,
    num_workers=4,               # DataLoader workers (0 for Windows)
    pin_memory=True,             # True for GPU training

    # Output
    checkpoint_dir="./checkpoints"
)
```

---

## Understanding the Results

The `run_simulation()` method returns a dictionary with:

```python
{
    'experiment_name': 'my_experiment',
    'num_clients': 5,
    'num_rounds': 10,
    'status': 'completed',
    'metrics': {
        'losses_distributed': [(round, loss), ...],
        'metrics_distributed': [metrics_dict, ...],
        'losses_centralized': [...],  # If centralized eval enabled
        'metrics_centralized': [...]
    }
}
```

### Accessing Metrics

```python
# Get final training loss
final_loss = results['metrics']['losses_distributed'][-1][1]

# Get loss per round
for round_num, (_, loss) in enumerate(results['metrics']['losses_distributed'], 1):
    print(f"Round {round_num}: Loss = {loss:.4f}")

# Get client metrics
for round_metrics in results['metrics']['metrics_distributed']:
    print(f"Client accuracies: {round_metrics}")
```

---

## Troubleshooting

### Issue: "history is None" or "AttributeError: 'NoneType' object has no attribute 'losses_distributed'"

**Cause**: Using `run_simulation` instead of `start_simulation`. `run_simulation` returns `None` in Flower 1.13+.

**Solution**: Use `start_simulation` which returns a `History` object:

```python
from flwr.simulation import start_simulation  # Correct

history = start_simulation(...)  # Returns History object
print(history.losses_distributed)  # Works!
```

Not this:
```python
from flwr.simulation import run_simulation  # Wrong for getting results

history = run_simulation(...)  # Returns None
print(history.losses_distributed)  # AttributeError!
```

### Issue: "Context has no attribute 'node_id'"

**Solution**: Ensure you're using Flower >= 1.13. The `Context` API changed in this version.

```bash
pip install --upgrade "flwr[simulation]>=1.13.0"
```

### Issue: Images not found during training

**Solution**: Ensure your image directory structure matches expectations:

```
image_dir/
├── patient_0001.png
├── patient_0002.png
└── ...
```

And your DataFrame has correct filenames in the filename column.

### Issue: Out of memory errors

**Solution**: Reduce resource allocation:

```python
config = ExperimentConfig(
    batch_size=8,           # Smaller batch size
    num_workers=0,          # Reduce workers
    clients_per_round=2     # Fewer concurrent clients
)

# In simulation_runner.py, adjust:
backend_config = {
    "client_resources": {
        "num_cpus": 1,
        "num_gpus": 0.0  # CPU only
    }
}
```

---

## Advanced: Custom Strategies

To use a custom strategy instead of FedAvg:

```python
from flwr.server.strategy import FedProx, FedOpt

# In simulation_runner.py, modify the server_fn:
def server_fn(context: Context) -> ServerAppComponents:
    strategy = FedProx(  # Or FedOpt, FedAdagrad, etc.
        fraction_fit=0.5,
        min_fit_clients=3,
        proximal_mu=0.1  # FedProx-specific parameter
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=10)
    )
```

---

## Next Steps

1. **Prepare Your Data**: Create metadata CSV with columns: `patientId`, `Target`, `filename`
2. **Configure Training**: Adjust `ExperimentConfig` for your use case
3. **Run Experiments**: Test different partitioning strategies
4. **Monitor Results**: Track metrics across rounds
5. **Save Models**: Final model is saved to `checkpoint_dir`

---

## References

- [Flower Documentation](https://flower.ai/docs/framework/)
- [Flower Simulation Guide](https://flower.ai/docs/framework/how-to-run-simulations.html)
- [Message API Migration](https://flower.ai/docs/framework/how-to-upgrade-to-flower-1.13.html)

---

## Support

For issues specific to this codebase, check:
- `SimulationRunner` implementation: `src/control/federated_learning/core/simulation_runner.py`
- `FlowerClient` implementation: `src/control/federated_learning/core/fed_client.py`
- Test examples: `tests/integration/federated_learning/`
