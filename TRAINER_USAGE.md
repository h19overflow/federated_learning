# Federated Trainer Orchestrator - Usage Guide

## Overview

The `FederatedTrainer` is a simplified orchestrator that coordinates the complete federated learning workflow from data preparation through model training to final checkpoint saving. It strictly follows SOLID principles with clear separation of concerns.

## Architecture

### Single Responsibility Principle

- **FederatedTrainer**: Orchestrates workflow only, NO data loading, NO training logic, NO strategy configuration
- **FlowerClient** (core.fed_client): Handles local training via fit() and evaluate()
- **ClientDataManager** (data.client_data): Creates DataLoaders for client partitions
- **partition_data_stratified** (data.partitioner): Partitions data across clients
- **ResNetWithCustomHead** (entities): Model definition
- **DataSourceExtractor & DatasetPreparer**: Data utilities

### Workflow

```
train() entry point
    ↓
Extract & validate data (DataSourceExtractor)
    ↓
Prepare dataset (DatasetPreparer)
    ↓
Partition data (partition_data_stratified)
    ↓
Create client dataloaders (ClientDataManager)
    ↓
Create client factory (client_fn closure)
    ↓
Initialize global model (ResNetWithCustomHead)
    ↓
Create strategy (FedAvg)
    ↓
Run simulation (start_simulation)
    ↓
Process results & save model
    ↓
Return results dict
```

## Usage Examples

### Basic Training

```python
from federated_pneumonia_detection.src.control.federated_learning.trainer import FederatedTrainer

# Initialize trainer with defaults
trainer = FederatedTrainer()

# Run federated training from a zip file
results = trainer.train(
    source_path="path/to/dataset.zip",
    experiment_name="pneumonia_fed_v1"
)

print(f"Training completed: {results['status']}")
print(f"Final loss: {results['metrics']['losses_distributed'][-1][1]:.4f}")
```

### Custom Configuration

```python
# Initialize with custom config and directories
trainer = FederatedTrainer(
    config_path="config/custom_config.yaml",
    checkpoint_dir="results/fed_checkpoints",
    logs_dir="results/fed_logs"
)

# Train from extracted directory
results = trainer.train(
    source_path="/path/to/extracted_dataset",
    experiment_name="pneumonia_fed_v2",
    csv_filename="metadata.csv"
)

# Access results
num_clients = results['num_clients']
num_rounds = results['num_rounds']
metrics = results['metrics']
```

### Working with Results

```python
results = trainer.train(
    source_path="dataset.zip",
    experiment_name="test_run"
)

# Distributed training metrics
losses = results['metrics']['losses_distributed']  # [(round, loss), ...]
metrics = results['metrics']['metrics_distributed']  # [[(client_metrics), ...], ...]

# Access saved model
checkpoint_path = f"{results['checkpoint_dir']}/test_run_final_model.pt"
```

## Key Methods

### `__init__(config_path, checkpoint_dir, logs_dir)`

Initializes the trainer with configuration and directories.

- Loads configuration (uses defaults if not specified)
- Creates checkpoint and log directories
- Initializes data utilities
- Sets up logging

**Parameters:**
- `config_path`: Optional path to YAML configuration file
- `checkpoint_dir`: Directory for saving model checkpoints (default: "fed_results/checkpoints")
- `logs_dir`: Directory for saving logs (default: "fed_results/logs")

### `train(source_path, experiment_name, csv_filename)`

Executes complete federated training workflow.

**Parameters:**
- `source_path`: Path to zip file or directory containing dataset
- `experiment_name`: Name for this training experiment
- `csv_filename`: Optional specific CSV filename to look for

**Returns:**
```python
{
    "experiment_name": str,           # Name of experiment
    "num_clients": int,               # Number of federated clients
    "num_rounds": int,                # Number of training rounds
    "status": str,                    # "completed" or error message
    "checkpoint_dir": str,            # Path to checkpoint directory
    "logs_dir": str,                  # Path to logs directory
    "metrics": {
        "losses_distributed": List,   # [(round, loss), ...]
        "metrics_distributed": List,  # Client metrics per round
        "losses_centralized": List,   # Server-side losses
        "metrics_centralized": List   # Server-side metrics
    }
}
```

## Internal Methods (Implementation Details)

### `_run_simulation(client_partitions, image_dir, experiment_name)`

Core simulation orchestrator. Coordinates the Flower simulation with strategy, dataloaders, and model.

### `_prepare_client_dataloaders(client_partitions, data_manager)`

Creates train/val DataLoaders for each client partition. Handles empty partitions gracefully.

### `_create_client_fn(client_dataloaders)`

Factory function generator that creates the `client_fn` closure for Flower. Each call to the returned function creates a FlowerClient with appropriate dataloaders.

**Key Feature:** Uses closure pattern to capture dataloaders without global state.

```python
def client_fn(context: Context):
    client_id = int(context.node_id)
    train_loader, val_loader = client_dataloaders[client_id]
    model = ResNetWithCustomHead(...)
    flower_client = FlowerClient(...)
    return flower_client.to_client()
```

### `_create_model()`

Creates a ResNetWithCustomHead instance with configuration parameters.

### `_create_strategy(initial_parameters, num_clients)`

Creates FedAvg strategy with proper fraction calculation and parameter initialization.

### `_process_results(history, experiment_name, num_clients)`

Formats Flower simulation history into structured results dictionary.

### `_save_final_model(history, experiment_name)`

Saves final trained model parameters to checkpoint file.

### `_setup_logging()`

Configures logger with formatting and stream handler.

## Configuration Parameters

Key configuration parameters from ExperimentConfig:

```yaml
experiment:
  # Federated learning
  num_rounds: 10               # Number of federation rounds
  num_clients: 5               # Number of total clients
  clients_per_round: 3         # Clients to sample per round
  local_epochs: 1              # Local training epochs per client

  # Model
  learning_rate: 0.001
  dropout_rate: 0.5
  fine_tune_layers_count: 0
  num_classes: 1               # Binary classification

  # Data
  validation_split: 0.2
  batch_size: 128
```

## Error Handling

The trainer handles errors at multiple levels:

1. **Configuration Loading**: Falls back to defaults if config file missing
2. **Data Extraction**: Validates zip/directory structure before processing
3. **DataLoader Creation**: Logs warnings for empty partitions, skips them
4. **Simulation**: Catches and logs all Flower simulation errors
5. **Model Saving**: Gracefully handles missing parameters

All errors are logged with context before re-raising.

## Performance Considerations

1. **Memory**: Large datasets are partitioned across clients to reduce memory footprint
2. **Data Augmentation**: Applied only to training data, validation uses no augmentation
3. **Workers**: DataLoaders use `num_workers=0` for Windows compatibility with Flower
4. **GPU Resources**: Configured as `num_gpus=0.0` - modify if GPUs available

## Comparison with Old FederatedTrainer

| Aspect | Old | New |
|--------|-----|-----|
| Lines of code | 407 | 395 |
| Single file | Yes | Yes |
| Global state | Yes (client_app_factory) | No (closure only) |
| Clear SRP | No (mixed concerns) | Yes (orchestration only) |
| Reusable components | Partial | Full |
| Type hints | Partial | Complete |
| Error handling | Basic | Comprehensive |
| Documentation | Minimal | Extensive |

## Testing

Basic smoke test:

```python
def test_federated_trainer_initialization():
    trainer = FederatedTrainer()
    assert trainer.config is not None
    assert trainer.constants is not None
    assert os.path.exists(trainer.checkpoint_dir)
    assert os.path.exists(trainer.logs_dir)

def test_client_fn_factory():
    # Mock dataloaders
    dataloaders = [(mock_train_loader, mock_val_loader)]
    trainer = FederatedTrainer()
    client_fn = trainer._create_client_fn(dataloaders)

    # Simulate context
    context = Mock(node_id="0")
    client = client_fn(context)
    assert client is not None
```

## File Structure

```
federated_pneumonia_detection/src/control/federated_learning/
├── trainer.py                 # NEW - Main orchestrator
├── core/
│   └── fed_client.py         # FlowerClient (local training)
├── data/
│   ├── client_data.py        # ClientDataManager
│   └── partitioner.py        # partition_data_stratified
├── training/
│   └── functions.py          # Training utilities
└── federated_trainer.py      # OLD - Will be deprecated
```

## Migration from Old FederatedTrainer

If migrating from the old `FederatedTrainer`, the new `trainer.py` provides the same public API:

```python
# Old code still works
from federated_pneumonia_detection.src.control.federated_learning.trainer import FederatedTrainer

trainer = FederatedTrainer()
results = trainer.train(source_path, experiment_name)
```

The main difference is internal: the new version uses cleaner separation of concerns without global state.
