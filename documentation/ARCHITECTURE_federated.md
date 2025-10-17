# Federated Learning Architecture

## Overview
This federated learning system implements a complete Flower-based FL pipeline for pneumonia detection. The code is organized by **functional responsibility** for intuitive navigation.

## Directory Structure

```
federated_learning/
├── federated_trainer.py          # Main entry point & orchestrator
│
├── core/                          # FL Infrastructure
│   ├── __init__.py
│   └── simulation_runner.py      # Flower simulation orchestration
│
├── data/                          # Data Management
│   ├── __init__.py
│   ├── partitioner.py            # Data partitioning strategies
│   └── client_data.py            # Client DataLoader creation
│
└── training/                      # Training Utilities
    ├── __init__.py
    └── functions.py              # Pure PyTorch training loops
```

## Component Responsibilities

### 1. FederatedTrainer (Entry Point)
**File**: `federated_trainer.py`
**Purpose**: Main orchestrator that coordinates the entire FL workflow

**Key Methods**:
- `train(source_path, experiment_name)` - Complete FL workflow from data to results
- `_partition_data_for_clients(df)` - Delegates to data partitioners
- `_run_federated_simulation()` - Delegates to SimulationRunner

**Example**:
```python
trainer = FederatedTrainer(
    partition_strategy='iid',
    checkpoint_dir='federated_checkpoints'
)
results = trainer.train('data.zip', 'experiment_1')
```

---

### 2. SimulationRunner (Core)
**File**: `core/simulation_runner.py`
**Purpose**: Bridges data partitions with Flower simulation

**Key Features**:
- Creates Flower clients with proper data access
- Initializes global model
- Configures FedAvg strategy
- Runs Flower simulation
- Saves final model

**How It Works**:
1. Takes data partitions + image directory
2. Creates DataLoaders for each client using `ClientDataManager`
3. Defines `client_fn` that creates `FlowerClient` instances
4. Runs Flower simulation with `run_simulation()`
5. Returns results and saves final model

---

### 3. Data Module (data/)

#### a) Partitioner (`partitioner.py`)
**Purpose**: Split datasets across clients using different strategies

**Functions**:
- `partition_data_iid()` - Random equal distribution
- `partition_data_by_patient()` - Patient-based (non-IID, realistic)
- `partition_data_stratified()` - Class-balanced distribution

**Example**:
```python
from data.partitioner import partition_data_stratified

partitions = partition_data_stratified(
    df=full_dataset,
    num_clients=10,
    target_column='Target',
    seed=42
)
```

#### b) ClientDataManager (`client_data.py`)
**Purpose**: Create train/val DataLoaders for individual clients

**Key Method**:
- `create_dataloaders_for_partition(partition_df)` - Returns (train_loader, val_loader)

**Features**:
- Handles train/val split with stratification
- Creates PyTorch datasets with transforms
- Configures DataLoaders with proper settings

---

### 4. Training Module (training/)

#### Functions (`functions.py`)
**Purpose**: Pure PyTorch training utilities (no Lightning, Flower-compatible)

**Key Functions**:
- `train_one_epoch()` - Single epoch training loop
- `evaluate_model()` - Model evaluation with metrics
- `get_model_parameters()` / `set_model_parameters()` - NumPy serialization
- `create_optimizer()` - AdamW optimizer creation

**Example**:
```python
from training.functions import train_one_epoch, create_optimizer

optimizer = create_optimizer(model, lr=0.001)
loss = train_one_epoch(model, train_loader, optimizer, device)
```

---

## Data Flow

```
1. FederatedTrainer.train(source_path)
   ↓
2. Extract/find data → Load CSV → Prepare DataFrame
   ↓
3. data.partitioner.partition_data_*() → List[DataFrame]
   ↓
4. SimulationRunner.run_simulation(partitions, image_dir)
   ↓
5. ClientDataManager.create_dataloaders_for_partition()
   ↓
6. FlowerClient instances with train/val DataLoaders
   ↓
7. training.functions.train_one_epoch() for local training
   ↓
8. Flower aggregates weights using FedAvg
   ↓
9. Results + saved model
```

## Key Design Principles

### 1. Find by Purpose
- Need data stuff? → `data/`
- Training functions? → `training/`
- FL infrastructure? → `core/`
- Entry point? → `federated_trainer.py`

### 2. Clear Dependencies
```
federated_trainer.py
    ↓
    uses: core.SimulationRunner
          data.partitioner

SimulationRunner
    ↓
    uses: data.ClientDataManager
          training.functions

ClientDataManager
    ↓
    uses: entities (CustomImageDataset, SystemConstants, etc.)

training.functions
    ↓
    uses: entities.ResNetWithCustomHead
```

### 3. Single Responsibility
- Each file has ONE clear job
- Max ~150 lines per file
- No "Manager" or "Handler" anti-patterns

### 4. Testability
- Components can be tested independently
- Clear interfaces between modules
- Dependency injection throughout

## Usage Examples

### Basic Training
```python
from federated_learning import FederatedTrainer

trainer = FederatedTrainer(partition_strategy='iid')
results = trainer.train('chest_xray_data.zip', 'experiment_1')
print(f"Final loss: {results['metrics']['losses_distributed'][-1]}")
```

### Custom Configuration
```python
trainer = FederatedTrainer(
    config_path='config/custom_fl_config.yaml',
    partition_strategy='non-iid',  # Patient-based
    checkpoint_dir='my_checkpoints',
    logs_dir='my_logs'
)

results = trainer.train(
    source_path='/path/to/data',
    experiment_name='patient_partitioned_experiment'
)
```

### Direct Simulation (Advanced)
```python
from core.simulation_runner import SimulationRunner
from data.partitioner import partition_data_stratified

# Prepare partitions
partitions = partition_data_stratified(df, num_clients=5, target_column='Target')

# Run simulation
runner = SimulationRunner(constants, config)
results = runner.run_simulation(partitions, image_dir, 'my_experiment')
```

## Migration from Old Structure

### Old Code
```python
# Old placeholder implementation
from federated_learning.data_partitioner import partition_data_iid
from federated_learning.training_functions import train_one_epoch

# Simulation was not implemented
```

### New Code
```python
# New organized structure
from federated_learning.data.partitioner import partition_data_iid
from federated_learning.training.functions import train_one_epoch
from federated_learning import FederatedTrainer

# Simulation is fully implemented
trainer = FederatedTrainer()
results = trainer.train('data.zip')  # Works end-to-end!
```

## What Changed

### Before (Placeholders)
- ❌ Flat structure, hard to navigate
- ❌ `client_app.py` had no data loading
- ❌ `federated_trainer._run_federated_simulation()` was a placeholder
- ❌ No bridge between partitions and Flower clients

### After (Fully Functional)
- ✅ Organized by functionality
- ✅ `SimulationRunner` implements complete Flower integration
- ✅ `ClientDataManager` creates actual DataLoaders from partitions
- ✅ `FlowerClient` trains with real data
- ✅ End-to-end working FL pipeline

## Next Steps

1. **Run a test**: Use the test script to verify the pipeline
2. **Customize strategies**: Try different partitioning strategies
3. **Monitor training**: Add TensorBoard or Weights & Biases integration
4. **Scale up**: Adjust `num_clients`, `num_rounds` in config
5. **Production deployment**: Use standalone Flower server/client apps

## Troubleshooting

**Q: Import errors?**
A: Ensure you're using the new paths:
```python
from federated_learning.data.partitioner import partition_data_iid  # ✅
from federated_learning.data_partitioner import partition_data_iid   # ❌ Old path
```

**Q: Simulation fails?**
A: Check:
- Data partitions are not empty
- Image directory exists and contains images
- Config has valid `num_clients`, `num_rounds`, `batch_size`

**Q: Out of memory?**
A: Reduce:
- `batch_size` in config
- `num_clients` (fewer concurrent clients)
- Set `client_resources = {"num_cpus": 1, "num_gpus": 0.0}` for CPU-only

---

**Architecture designed for clarity, maintainability, and scalability.**
