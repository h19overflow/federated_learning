# Federated Learning Implementation Summary

## Overview
Simplified federated learning system following the plan.md specification and Flower framework best practices. The implementation directly aligns with the documented architecture while leveraging existing components.

## Architecture

### Components Structure

```
federated_learning/
├── client.py              # FlowerClient implementation
├── data_manager.py        # Client data loading and preprocessing
├── partitioner.py         # Data partitioning across clients
├── trainer.py             # FederatedTrainer orchestration (NEW)
├── plan.md                # Architecture specification
└── __init__.py            # Module exports
```

## Implementation Details

### 1. FlowerClient (`client.py`)
**Purpose:** Individual federated learning participant

**Key Methods:**
- `get_parameters()`: Extract model weights as NumPy arrays
- `set_parameters()`: Load weights from server
- `fit()`: Local training on client data
- `evaluate()`: Local validation on client data

**Dependencies:**
- Uses `ResNetWithCustomHead` for binary classification (pneumonia detection)
- Supports both BCEWithLogitsLoss (binary) and CrossEntropyLoss (multiclass)
- Integrates with PyTorch Lightning Trainer for training/evaluation

### 2. Data Manager (`data_manager.py`)
**Purpose:** Convert data partitions into PyTorch DataLoaders

**Key Functions:**
- `load_data()`: Create train/val DataLoaders from client partition
- `split_partition()`: Stratified train/val splitting with fallback to random split

**Features:**
- Applies image augmentation and normalization
- Handles Windows/Flower compatibility (num_workers=0)
- Validates images and manages transforms

### 3. Partitioner (`partitioner.py`)
**Purpose:** Distribute data fairly across clients

**Key Functions:**
- `partition_data_stratified()`: IID partitioning maintaining class balance
  - Splits each class independently
  - Distributes proportionally to each client
  - Ensures balanced class representation

### 4. FederatedTrainer (`trainer.py`) - NEW
**Purpose:** Orchestrate the entire FL simulation

**Workflow:**

```
1. Data Partitioning
   ├─ Partition training data across clients (IID with stratification)
   └─ Log class distribution per client

2. DataLoader Creation
   ├─ Create train/val loaders for each client
   └─ Cache for client factory function

3. Model Initialization
   ├─ Create initial ResNetWithCustomHead
   └─ Extract parameters as Flower Parameters

4. Strategy Configuration
   ├─ FedAvg strategy with 100% client participation
   ├─ Server-side evaluation on reserved validation set
   └─ No client-side evaluation (fraction_evaluate=0.0)

5. Flower Simulation
   ├─ start_simulation with client_fn factory
   ├─ Each round: clients train locally, server aggregates
   └─ Global evaluation after each round

6. Results Collection
   └─ Return history with losses and metrics per round
```

**Key Methods:**
- `_create_model()`: Instantiate ResNetWithCustomHead
- `_get_initial_parameters()`: Extract initial weights for server
- `_client_fn()`: Factory to create FlowerClient instances
- `_create_evaluate_fn()`: Server-side evaluation on global validation set
- `train()`: Main orchestration method

**Key Features:**
- Clean separation of concerns
- Comprehensive logging at each stage
- Error handling with detailed error messages
- Minimal configuration required

## Data Flow

```
Training Metadata CSV
     ↓
load_metadata() → DataFrame
     ↓
partition_data_stratified() → [Client0_df, Client1_df, ...]
     ↓
load_data() for each partition → [(train_loader, val_loader), ...]
     ↓
FlowerClient with dataloaders
     ↓
FedAvg Strategy (Aggregation)
     ↓
Global Model Evaluation
     ↓
Results & History
```

## Usage

### Quick Start
```python
from pathlib import Path
import torch
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.control.federated_learning import FederatedTrainer
from federated_pneumonia_detection.src.utils.data_processing import load_metadata

# Setup
constants = SystemConstants()
config = ExperimentConfig(
    num_clients=2,
    num_rounds=5,
    local_epochs=2,
    learning_rate=0.001
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data_df = load_metadata("Training/stage2_train_metadata.csv", constants)

# Train
trainer = FederatedTrainer(config, constants, device)
results = trainer.train(
    data_df=data_df,
    image_dir=Path("Training/Images"),
    experiment_name="pneumonia_federated"
)
```

### Run Script
```bash
python run_federated_training.py
```

## Configuration

### ExperimentConfig Parameters
```python
# Federated Learning specific
num_clients: int = 5              # Number of federated clients
num_rounds: int = 10              # Number of FL rounds
local_epochs: int = 1             # Epochs per client per round
clients_per_round: int = 3        # (Optional for advanced strategies)

# Model
learning_rate: float = 0.001
weight_decay: float = 0.0001
dropout_rate: float = 0.5
num_classes: int = 1              # Binary classification (pneumonia)

# Data
batch_size: int = 128
validation_split: float = 0.20
seed: int = 42
```

## Alignment with Plan.md

✅ **FlowerClient Implementation**
- Initializes with client ID, data partitions, and image directory
- Computes local class weights
- Instantiates LitResNet model
- Sets up XRayDataModule for data loading
- get_parameters() retrieves model weights
- set_parameters() updates from server
- fit() trains locally and returns updated parameters
- evaluate() validates locally

✅ **FedAvg Strategy Integration**
- fraction_fit=1.0: All clients participate in training
- fraction_evaluate=0.0: No client evaluation (server only)
- Server-side evaluate_fn for global validation
- Initial parameters from temporary model instance
- FedAvg aggregation combining weighted client updates

✅ **Simulation Process**
1. Server selects clients and sends global model
2. Clients receive weights and train locally
3. Clients return updated parameters and dataset size
4. Server aggregates using FedAvg (weighted averaging)
5. Server evaluates global model on validation set
6. Process repeats for each round

✅ **Result Tracking**
- History object with metrics per round
- Validation loss and accuracy after each round
- Training loss from client-side training
- Final performance metrics

## Key Design Decisions

1. **Simple & Minimal**: No complex abstractions, direct Flower API usage
2. **Existing Components**: Leverages FlowerClient, data_manager, partitioner
3. **IID Data**: Stratified partitioning maintains class balance across clients
4. **Server-Side Evaluation**: More reliable than client-side for unbiased metrics
5. **Class Weights**: Handled in FlowerClient for local training
6. **Device Agnostic**: Works with CPU, GPU (CUDA), and MPS

## Performance Considerations

- **Windows Compatibility**: num_workers=0 for DataLoaders
- **Memory**: Models created fresh per client/evaluation
- **Scalability**: Client factory pattern enables many clients
- **Reproducibility**: Seeded partitioning and training

## Testing the Implementation

Verify imports work:
```bash
python -c "from federated_pneumonia_detection.src.control.federated_learning import FederatedTrainer; print('OK')"
```

Check module compiles:
```bash
python -m py_compile federated_pneumonia_detection/src/control/federated_learning/trainer.py
```

## Next Steps

1. Run training: `python run_federated_training.py`
2. Monitor logs for each FL round
3. Compare results with centralized training baseline
4. Adjust hyperparameters as needed
5. Export final model from best round

## File Changes Summary

| File | Change |
|------|--------|
| `trainer.py` | Created - Main orchestration class |
| `__init__.py` | Updated - Export FederatedTrainer |
| `run_federated_training.py` | Updated - Use new FederatedTrainer |

All changes are minimal, focused, and follow the existing code patterns and guidelines.
