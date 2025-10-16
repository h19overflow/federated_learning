# Training Scripts - Quick Start Guide

Three scripts to train and compare pneumonia detection models using centralized and federated learning approaches.

## Prerequisites

- Python environment with project dependencies installed
- Dataset in `Training/` folder with structure:
  - `Training/Images/` - chest X-ray images
  - `Training/stage2_train_metadata.csv` - metadata file

## Scripts Overview

### 1. Centralized Training (`run_centralized_training.py`)

Trains a single model on all data using PyTorch Lightning.

**Run:**
```powershell
python -m run_centralized_training
```

**Configuration:**
- Epochs: 15
- Learning rate: 0.0015
- Batch size: 512
- Validation split: 20%

**Outputs:**
- Checkpoints: `results/centralized/checkpoints/`
- Logs: `results/centralized/logs/`

### 2. Federated Training (`run_federated_training.py`)

Trains using federated learning simulation with multiple clients.

**Run:**
```powershell
python -m run_federated_training
```

**Configuration:**
- Rounds: 15
- Clients: 5
- Clients per round: 3
- Local epochs: 3
- Partition strategy: stratified
- Learning rate: 0.0015
- Batch size: 512

**Outputs:**
- Checkpoints: `results/federated/checkpoints/`
- Logs: `results/federated/logs/`

### 3. Comparison Experiment (`run_comparison_experiment.py`)

Runs both centralized and federated training, then generates comparison report.

**Run:**
```powershell
python -m run_comparison_experiment
```

**Outputs:**
- Timestamped experiment directory: `experiments/experiment_YYYYMMDD_HHMMSS/`
  - `centralized/` - centralized training results
  - `federated/` - federated training results
  - Comparison metrics and reports

## Customization

To modify training parameters, edit the configuration variables at the top of each script:

```python
# Example: Change partition strategy in run_federated_training.py
partition_strategy = "stratified"  # Options: 'iid', 'stratified', 'by_patient'

# Example: Change output directory
checkpoint_dir = "results/federated/checkpoints"
```

## Data Requirements

The `Training/` folder must contain:
- **Images subfolder**: Contains .dcm or image files referenced by metadata
- **CSV file**: `stage2_train_metadata.csv` with columns:
  - patientId
  - x, y, width, height (bounding box coordinates)
  - Target (0 or 1 indicating pneumonia presence)
  - class (Normal/No Lung Opacity / Not Normal / Lung Opacity)
  - age, sex, modality, position (patient metadata)

## Expected Runtime

- **Centralized**: 15-30 minutes (depends on hardware)
- **Federated**: 20-40 minutes (depends on number of clients/rounds)
- **Comparison**: 35-70 minutes (runs both sequentially)

## Troubleshooting

**Issue**: `FileNotFoundError: Training directory not found`
- Ensure `Training/` folder exists in project root
- Verify `Training/Images/` and `Training/stage2_train_metadata.csv` are present

**Issue**: `CUDA out of memory`
- Reduce batch_size in script configuration
- Set GPU fraction in federated learning: `"num_gpus": 0.5`

**Issue**: Script exits without error
- Check logs in respective `logs/` directories
- Verify CSV file has correct column names

## Architecture

All scripts use the same underlying components:
- **DataSourceExtractor**: Validates and loads dataset
- **DatasetPreparer**: Processes data and creates train/val splits
- **ConfigLoader**: Manages system configuration and hyperparameters

The scripts differ only in training approach:
- Centralized uses `CentralizedTrainer` with PyTorch Lightning
- Federated uses `FederatedTrainer` with Flower framework
- Comparison uses `ExperimentOrchestrator` to run and compare both
