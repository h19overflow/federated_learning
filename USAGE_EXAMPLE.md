# Federated Learning Integration - Usage Examples

This guide shows how to use the newly integrated federated learning system for pneumonia detection.

## Overview

The system now supports three modes:
1. **Centralized Training**: Train on all data in one location (existing)
2. **Federated Learning**: Train across distributed clients
3. **Comparison Mode**: Run both and compare results

## Quick Start

### Option 1: Run Comparison (Recommended)

Run both centralized and federated training, then compare:

```python
from federated_pneumonia_detection.src.control.comparison import run_quick_comparison

# Simple one-liner for complete comparison
results = run_quick_comparison(
    source_path="path/to/dataset.zip",  # or path to directory
    config_path="federated_pneumonia_detection/config/default_config.yaml",  # optional
    partition_strategy="iid"  # 'iid', 'non-iid', or 'stratified'
)

print(f"Results saved to: {results['experiment_dir']}")
```

### Option 2: Centralized Training Only

```python
from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer

trainer = CentralizedTrainer(
    config_path="federated_pneumonia_detection/config/default_config.yaml",
    checkpoint_dir="centralized_checkpoints",
    logs_dir="centralized_logs"
)

results = trainer.train(
    source_path="path/to/dataset.zip",
    experiment_name="my_centralized_experiment"
)

print(f"Best model: {results['best_model_path']}")
print(f"Final metrics: {results['final_metrics']}")
```

### Option 3: Federated Learning Only

```python
from federated_pneumonia_detection.src.control.federated_learning.federated_trainer import FederatedTrainer

trainer = FederatedTrainer(
    config_path="federated_pneumonia_detection/config/default_config.yaml",
    checkpoint_dir="federated_checkpoints",
    logs_dir="federated_logs",
    partition_strategy="iid"  # Data partitioning strategy
)

results = trainer.train(
    source_path="path/to/dataset.zip",
    experiment_name="my_federated_experiment"
)

print(f"Federated training completed: {results}")
```

### Option 4: Advanced Orchestration

For full control over comparison experiments:

```python
from federated_pneumonia_detection.src.control.comparison import ExperimentOrchestrator

# Create orchestrator
orchestrator = ExperimentOrchestrator(
    config_path="federated_pneumonia_detection/config/default_config.yaml",
    base_output_dir="my_experiments",
    partition_strategy="stratified"
)

# Run individual experiments
centralized_results = orchestrator.run_centralized("path/to/dataset.zip")
federated_results = orchestrator.run_federated("path/to/dataset.zip")

# Or run full comparison
comparison = orchestrator.run_comparison("path/to/dataset.zip")

print(f"Experiment directory: {orchestrator.experiment_dir}")
```

## Configuration

All parameters are controlled via `config/default_config.yaml`:

### Key Federated Learning Parameters:

```yaml
experiment:
  # Federated Learning parameters
  num_rounds: 15              # Number of FL rounds
  num_clients: 5              # Total number of clients
  clients_per_round: 3        # Clients participating per round
  local_epochs: 2             # Epochs each client trains locally

  # Model parameters (shared by both centralized and federated)
  learning_rate: 0.0015
  epochs: 15                  # For centralized training
  batch_size: 512
  dropout_rate: 0.3
```

## Data Partitioning Strategies

### IID (Independent and Identically Distributed)
- Random distribution of data across clients
- Each client gets similar data distribution

```python
trainer = FederatedTrainer(partition_strategy="iid")
```

### Non-IID (Patient-based)
- Each client gets data from distinct patients
- More realistic for medical scenarios

```python
trainer = FederatedTrainer(partition_strategy="non-iid")
```

### Stratified
- Maintains class balance across clients
- Good for imbalanced datasets

```python
trainer = FederatedTrainer(partition_strategy="stratified")
```

## Input Data Formats

The system accepts:
1. **ZIP files**: Containing `Images/` folder and `Train_metadata.csv`
2. **Directories**: With the same structure

### Expected Structure:
```
dataset/
├── Images/
│   └── Images/
│       ├── patient_001.png
│       ├── patient_002.png
│       └── ...
└── Train_metadata.csv
```

### CSV Format:
```csv
patientId,Target
patient_001,0
patient_002,1
```

## Output Structure

When running comparison experiments:

```
experiments/
└── experiment_20250101_120000/
    ├── centralized/
    │   ├── checkpoints/
    │   │   └── pneumonia_model.ckpt
    │   └── logs/
    │       └── version_0/
    ├── federated/
    │   ├── checkpoints/
    │   │   └── federated_final_model.pt
    │   └── logs/
    ├── centralized_results.json
    ├── federated_results.json
    └── comparison_report.json
```

## Architecture Highlights

### Reused Components ✅
- `ConfigLoader`: YAML configuration loading
- `ResNetWithCustomHead`: Same model for both approaches
- `CustomImageDataset`: Dataset handling
- `TransformBuilder`: Image preprocessing
- `data_processing.py`: Metadata and splitting utilities

### New Components 🆕
- `data_partitioner.py`: Splits data across clients
- `training_functions.py`: Pure PyTorch training (Flower-compatible)
- `federated_trainer.py`: FL orchestrator (mirrors CentralizedTrainer)
- `experiment_orchestrator.py`: Comparison system

### Consistent API
Both trainers follow the same interface:

```python
trainer = Trainer(config_path, checkpoint_dir, logs_dir)
results = trainer.train(source_path, experiment_name)
```

## Example Workflow

```python
from federated_pneumonia_detection.src.control.comparison import ExperimentOrchestrator

# 1. Initialize orchestrator
orchestrator = ExperimentOrchestrator(
    config_path="federated_pneumonia_detection/config/default_config.yaml",
    partition_strategy="non-iid"  # Patient-based partitioning
)

# 2. Run comparison experiment
results = orchestrator.run_comparison(
    source_path="datasets/chest_xray.zip",
    centralized_name="baseline_centralized",
    federated_name="privacy_preserving_fl"
)

# 3. Check results
if results['centralized']['status'] == 'success':
    print("✓ Centralized training succeeded")

if results['federated']['status'] == 'success':
    print("✓ Federated learning succeeded")

# 4. Results are saved automatically
print(f"Full report: {orchestrator.experiment_dir}/comparison_report.json")
```

## Next Steps

### For Full Flower Integration:
The current implementation provides the foundation. To complete the Flower simulation:

1. Implement data loading in `client_app.py` train/evaluate functions
2. Set up Flower simulation in `federated_trainer.py._run_federated_simulation()`
3. Use `flwr.simulation.start_simulation()` with client functions

### For Visualization:
Create visualization tools in `src/control/comparison/results_comparator.py` to:
- Plot training curves (centralized vs federated)
- Compare final metrics (accuracy, loss, F1, etc.)
- Generate comparison charts

## Configuration Tips

### For Quick Experiments:
```yaml
system:
  sample_fraction: 0.10  # Use 10% of data

experiment:
  epochs: 5              # Centralized
  num_rounds: 5          # Federated
  num_clients: 3
```

### For Production:
```yaml
system:
  sample_fraction: 1.0   # Full dataset

experiment:
  epochs: 15
  num_rounds: 15
  num_clients: 5
  clients_per_round: 3
```

## Troubleshooting

### Issue: Configuration not loading
**Solution**: Ensure `config_dir` path is relative to project root:
```python
config_loader = ConfigLoader(config_dir="federated_pneumonia_detection/config")
```

### Issue: Data partitioning fails
**Solution**: Check that dataset has enough samples for the number of clients:
```python
# Ensure: len(dataset) >= num_clients
```

### Issue: Model architecture mismatch
**Solution**: Both trainers use the same `ResNetWithCustomHead` - ensure configuration is consistent.

## Support

For issues or questions:
1. Check configuration in `config/default_config.yaml`
2. Review logs in experiment output directories
3. Verify data format matches expected structure
