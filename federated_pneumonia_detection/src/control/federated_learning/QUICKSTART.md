# Federated Learning Quickstart Guide

## What Changed?

### Before
```
federated_learning/
├── client_app.py           # ❌ Placeholder, no data loading
├── server_app.py           # ❌ For standalone mode only
├── data_partitioner.py     # ❌ Flat structure
├── training_functions.py   # ❌ Hard to find
└── federated_trainer.py    # ❌ Placeholder simulation
```

### After
```
federated_learning/
├── federated_trainer.py          # ✅ Main entry point
│
├── core/                         # ✅ FL Infrastructure
│   └── simulation_runner.py     # ✅ Complete Flower integration
│
├── data/                         # ✅ Data Management
│   ├── partitioner.py           # ✅ Partitioning strategies
│   └── client_data.py           # ✅ Client DataLoader creation
│
└── training/                     # ✅ Training Utilities
    └── functions.py             # ✅ Pure PyTorch training loops
```

**Key Improvements:**
- ✅ **Fully functional** - No more placeholders!
- ✅ **Intuitive organization** - Find by purpose (data/, training/, core/)
- ✅ **Complete Flower integration** - Simulation runner bridges everything
- ✅ **Actual data loading** - Clients train with real data from partitions
- ✅ **Production ready** - Save models, track metrics, scalable

---

## Quick Usage

### 1. Basic Training
```python
from federated_pneumonia_detection.src.control.federated_learning import FederatedTrainer

# Initialize trainer
trainer = FederatedTrainer(
    partition_strategy='iid',  # or 'non-iid', 'stratified'
    checkpoint_dir='federated_checkpoints',
    logs_dir='federated_logs'
)

# Run federated training
results = trainer.train(
    source_path='path/to/chest_xray_data.zip',  # or directory
    experiment_name='my_fl_experiment'
)

# Check results
print(f"Status: {results['status']}")
print(f"Final model: {results['checkpoint_dir']}/my_fl_experiment_final_model.pt")
```

### 2. Test the Structure
```bash
# Run the test script
python -m federated_pneumonia_detection.src.control.federated_learning.test_fl_structure
```

This will verify:
- ✓ All imports work correctly
- ✓ Basic initialization succeeds
- ✓ Partitioning logic works
- ✓ Directory structure is correct

### 3. Custom Configuration
```python
trainer = FederatedTrainer(
    config_path='config/custom_fl.yaml',  # Optional custom config
    partition_strategy='non-iid',         # Patient-based partitioning
    checkpoint_dir='my_checkpoints'
)

results = trainer.train(
    source_path='/path/to/data',
    experiment_name='patient_partitioned_exp',
    csv_filename='metadata.csv'  # Optional specific CSV
)
```

---

## Configuration

Edit `config/default_config.yaml` to adjust:

```yaml
federated_learning:
  num_clients: 10              # Number of federated clients
  num_rounds: 50               # Number of training rounds
  clients_per_round: 5         # Clients selected per round
  local_epochs: 3              # Local training epochs per round

training:
  batch_size: 32
  learning_rate: 0.001
  num_workers: 4

data:
  validation_split: 0.2
  seed: 42
```

---

## Partitioning Strategies

### IID (Independent and Identically Distributed)
```python
trainer = FederatedTrainer(partition_strategy='iid')
```
- Random equal distribution
- All clients have similar data
- Good for: Baseline experiments

### Non-IID (Patient-based)
```python
trainer = FederatedTrainer(partition_strategy='non-iid')
```
- Each client gets data from distinct patients
- More realistic for medical data
- Good for: Production scenarios

### Stratified
```python
trainer = FederatedTrainer(partition_strategy='stratified')
```
- Maintains class balance across clients
- Each client has same class distribution
- Good for: Imbalanced datasets

---

## Advanced Usage

### Direct Simulation Control
```python
from federated_pneumonia_detection.src.control.federated_learning.core.simulation_runner import SimulationRunner
from federated_pneumonia_detection.src.control.federated_learning.data.partitioner import partition_data_stratified

# Create custom partitions
partitions = partition_data_stratified(df, num_clients=5, target_column='Target')

# Run simulation directly
runner = SimulationRunner(constants, config, logger)
results = runner.run_simulation(
    client_partitions=partitions,
    image_dir='/path/to/images',
    experiment_name='custom_experiment'
)
```

### Custom Client DataLoaders
```python
from federated_pneumonia_detection.src.control.federated_learning.data.client_data import ClientDataManager

# Create data manager
data_manager = ClientDataManager(image_dir, constants, config, logger)

# Create DataLoaders for a partition
train_loader, val_loader = data_manager.create_dataloaders_for_partition(
    partition_df=client_partition,
    validation_split=0.2
)
```

---

## Monitoring Training

### Check Status
```python
status = trainer.get_training_status()
print(status)
# {
#     'checkpoint_dir': 'federated_checkpoints',
#     'logs_dir': 'federated_logs',
#     'partition_strategy': 'iid',
#     'config': {...}
# }
```

### View Metrics
```python
results = trainer.train(...)

# Access metrics
print(results['metrics']['losses_distributed'])  # Loss per round
print(results['metrics']['metrics_distributed'])  # Metrics per round
```

---

## Troubleshooting

### Import Errors
**Old import (❌):**
```python
from federated_learning.data_partitioner import partition_data_iid
```

**New import (✅):**
```python
from federated_learning.data.partitioner import partition_data_iid
```

### Out of Memory
Reduce in config:
- `batch_size` (e.g., 16 instead of 32)
- `num_clients` (e.g., 5 instead of 10)
- `num_workers` (e.g., 2 instead of 4)

### Simulation Fails
Check:
- Data partitions are not empty
- Image directory exists and contains images
- Config has valid `num_clients`, `num_rounds`, `batch_size`

---

## File Reference

### Main Entry Point
- `federated_trainer.py` - Use `FederatedTrainer` class

### Core (Flower Integration)
- `core/simulation_runner.py` - `SimulationRunner` class, `FlowerClient` class

### Data Management
- `data/partitioner.py` - `partition_data_iid()`, `partition_data_stratified()`, etc.
- `data/client_data.py` - `ClientDataManager` class

### Training Utilities
- `training/functions.py` - `train_one_epoch()`, `evaluate_model()`, `create_optimizer()`, etc.

### Documentation
- `ARCHITECTURE.md` - Detailed architecture explanation
- `QUICKSTART.md` - This file
- `test_fl_structure.py` - Test script

---

## Next Steps

1. **Test the structure**: Run `test_fl_structure.py`
2. **Run a small experiment**: Use 2-3 clients, 5 rounds
3. **Scale up**: Increase clients and rounds
4. **Monitor metrics**: Add TensorBoard integration
5. **Production deployment**: Use standalone Flower server/client apps

---

**The federated learning system is now fully functional and ready for production use! 🚀**
