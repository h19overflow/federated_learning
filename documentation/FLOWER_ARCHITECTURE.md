# Federated Learning with Flower - Complete Architecture Guide

## Overview

This document explains the complete federated learning architecture using Flower's modern ClientApp pattern.

## Key Components

### 1. FlowerClient (fed_client.py)

**Location**: `federated_pneumonia_detection/src/control/federated_learning/core/fed_client.py`

The `FlowerClient` class is the core of your federated learning implementation. It extends `NumPyClient` from Flower.

```python
class FlowerClient(NumPyClient):
    """Flower NumPy client that performs local training."""
    
    def __init__(self, client_id, train_loader, val_loader, constants, config, logger):
        # Initialize with client-specific data and model
        self.model = ResNetWithCustomHead(...)
        
    def get_parameters(self, config) -> List:
        """Return current model parameters as NumPy arrays."""
        return get_model_parameters(self.model)
    
    def set_parameters(self, parameters: List):
        """Set model parameters from server."""
        set_model_parameters(self.model, parameters)
    
    def fit(self, parameters, config) -> Tuple[List, int, Dict]:
        """
        Train model on local data.
        1. Receives parameters from server
        2. Trains locally for N epochs
        3. Returns updated parameters + metrics
        """
        self.set_parameters(parameters)
        # ... train model ...
        return updated_parameters, num_examples, metrics
    
    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local validation data.
        Returns: (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)
        # ... evaluate model ...
        return loss, num_examples, metrics
```

**Important**: `FlowerClient` extends `NumPyClient`, which provides the `.to_client()` method. This method converts the NumPyClient to a Client instance that Flower can use.

### 2. FederatedTrainer (federated_trainer.py)

**Location**: `federated_pneumonia_detection/src/control/federated_learning/federated_trainer.py`

This orchestrates the entire federated learning workflow.

#### Key Methods:

**`train(source_path)`** - Main entry point
```python
trainer = FederatedTrainer()
results = trainer.train("dataset.zip")
```

Flow:
1. Extract/load dataset
2. Partition data across clients
3. Run federated simulation
4. Return results

**`_run_federated_simulation()`** - Core FL logic

This method implements the ClientApp pattern:

```python
def _run_federated_simulation(self, client_partitions, image_dir, experiment_name):
    # 1. Create data loaders for each client
    client_dataloaders = []
    for partition in client_partitions:
        train_loader, val_loader = create_dataloaders(partition)
        client_dataloaders.append((train_loader, val_loader))
    
    # 2. Define client factory function
    def client_fn(context: Context):
        """Creates a FlowerClient and converts to Client."""
        client_id = int(context.node_id)
        train_loader, val_loader = client_dataloaders[client_id]
        
        # Create FlowerClient (NumPyClient)
        flower_client = FlowerClient(
            client_id=client_id,
            train_loader=train_loader,
            val_loader=val_loader,
            constants=self.constants,
            config=self.config,
            logger=self.logger
        )
        
        # Convert to Client using inherited .to_client() method
        return flower_client.to_client()
    
    # 3. Create ClientApp (modern Flower pattern)
    client_app = ClientApp(client_fn=client_fn)
    
    # 4. Create global model and strategy
    global_model = ResNetWithCustomHead(...)
    initial_parameters = get_model_parameters(global_model)
    
    strategy = FedAvg(
        initial_parameters=initial_parameters,
        min_fit_clients=config.clients_per_round,
        # ... other strategy params
    )
    
    # 5. Run simulation
    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    
    return results
```

## How It Works - Step by Step

### Initialization Phase

```
FederatedTrainer.__init__()
    ├── Load configuration (SystemConstants, ExperimentConfig)
    ├── Setup directories (checkpoints, logs)
    └── Initialize data handlers
```

### Training Phase

```
trainer.train("dataset.zip")
    │
    ├── 1. Data Preparation
    │   ├── Extract/validate dataset
    │   ├── Load images and labels
    │   └── Partition across N clients (stratified)
    │
    ├── 2. Setup Federated Learning
    │   ├── Create data loaders for each client
    │   ├── Define client_fn factory
    │   ├── Create ClientApp
    │   ├── Initialize global model
    │   └── Configure FedAvg strategy
    │
    ├── 3. Run Simulation
    │   └── start_simulation() - for each round:
    │       │
    │       ├── SERVER: Select clients for this round
    │       │
    │       ├── For each selected client:
    │       │   ├── Call client_fn(context) → creates Client
    │       │   ├── Send global parameters to client
    │       │   ├── CLIENT: fit() → local training
    │       │   └── CLIENT: returns updated parameters
    │       │
    │       ├── SERVER: Aggregate client parameters (FedAvg)
    │       │
    │       └── SERVER: evaluate() (optional)
    │
    └── 4. Post-Processing
        ├── Process metrics from history
        ├── Save final model
        └── Return results
```

### What Happens in Each Round

```
Round N:
    1. Server selects K clients (e.g., 3 out of 5)
    
    2. For Client 0:
       ├── client_fn(context) creates FlowerClient instance
       ├── fit() is called with global parameters
       ├── Local training: 3 epochs on client's data
       └── Returns updated parameters + loss
    
    3. For Client 1:
       ├── (same process)
       └── ...
    
    4. For Client 2:
       ├── (same process)
       └── ...
    
    5. Server aggregates:
       new_params = weighted_average([client0_params, client1_params, client2_params])
    
    6. Global model updated with new_params
    
    7. (Optional) Server evaluation on centralized test set
```

## The ClientApp Pattern Explained

### Why ClientApp?

Flower 1.0+ introduced `ClientApp` for a clean separation of concerns:

- **Client logic** (FlowerClient): What to do during training/evaluation
- **Client creation** (client_fn): How to create clients with their data
- **Client app** (ClientApp): Wrapper that Flower framework understands

### The .to_client() Method

`NumPyClient.to_client()` is inherited from Flower's `NumPyClient` base class. It:

1. Wraps your NumPyClient in a Client instance
2. Handles serialization/deserialization of numpy arrays
3. Manages communication protocol with server

You don't implement it - it's automatically available when you extend `NumPyClient`.

```python
# In FlowerClient (extends NumPyClient)
# .to_client() is inherited - you don't need to define it!

# Usage in client_fn:
flower_client = FlowerClient(...)  # NumPyClient instance
client = flower_client.to_client()  # Converts to Client
return client  # Flower needs Client, not NumPyClient
```

## Configuration

### SystemConstants
- Dataset paths
- Image dimensions
- Model architecture settings

### ExperimentConfig
- `num_clients`: Total number of federated clients (e.g., 5)
- `clients_per_round`: How many participate each round (e.g., 3)
- `num_rounds`: Number of federated rounds (e.g., 10)
- `local_epochs`: Epochs each client trains locally (e.g., 3)
- `learning_rate`: Client learning rate
- `batch_size`: Client batch size

## Data Flow

```
Dataset (ZIP)
    │
    ├── Extract → Images + CSV
    │
    ├── Prepare → DataFrame with labels
    │
    ├── Partition → Split across N clients
    │   ├── Client 0: 200 samples (stratified)
    │   ├── Client 1: 200 samples
    │   └── ...
    │
    ├── Create DataLoaders → PyTorch DataLoader per client
    │   ├── Client 0: train_loader, val_loader
    │   ├── Client 1: train_loader, val_loader
    │   └── ...
    │
    └── Federated Training
        └── Each client trains on its local data only
```

## Example Usage

```python
from federated_pneumonia_detection.src.control.federated_learning import FederatedTrainer

# Initialize trainer
trainer = FederatedTrainer(
    checkpoint_dir="results/checkpoints",
    logs_dir="results/logs",
    partition_strategy="stratified"
)

# Run federated training
results = trainer.train(
    source_path="data/chest_xray.zip",
    experiment_name="federated_pneumonia_v1"
)

# Results contain:
# - Training history (losses, metrics per round)
# - Final model path
# - Client statistics
```

## Key Differences from Centralized Training

| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| Data | All data in one place | Distributed across clients |
| Training | One model, one dataset | Multiple local models → aggregated |
| Privacy | Data centralized | Data stays at clients |
| Model | Trained on full dataset | Trained on local data, averaged |
| Rounds | Epochs | Federated rounds × local epochs |

## Debugging Tips

1. **Check client creation**: Add logging in `client_fn` to verify clients are created
2. **Verify data loaders**: Ensure each client has non-empty data
3. **Monitor parameters**: Log parameter shapes in `fit()` and `evaluate()`
4. **Start small**: Test with 2 clients, 2 rounds first
5. **Check history**: Inspect `history.losses_distributed` for training progress

## Common Issues

### "Client ID out of range"
- More clients requested than data loaders created
- Check `num_clients` matches partitions

### ".to_client() not found"
- Ensure FlowerClient extends NumPyClient
- Method is inherited, not implemented by you

### "Empty dataloaders"
- Partition too small for batch size
- Check partition sizes vs batch_size

## References

- Flower Documentation: https://flower.ai/docs/
- ClientApp API: https://flower.ai/docs/framework/ref-api/flwr.client.ClientApp.html
- NumPyClient: https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html
