# How to Use the Database Save Feature

## Automatic Usage (Recommended)

The database save is now **automatic** - no code changes needed to your training scripts. Just run training normally:

```python
from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer

# Initialize trainer as usual
trainer = CentralizedTrainer(
    config_path=None,
    checkpoint_dir="results/centralized/checkpoints",
    logs_dir="results/centralized/logs"
)

# Run training - database save happens automatically
results = trainer.train(
    source_path="Training",
    experiment_name="pneumonia_centralized",
    csv_filename="stage2_train_metadata.csv"
)

# Access the saved run ID
print(f"Training run saved with ID: {results['run_id']}")
```

## What Gets Saved Automatically

When you run training, these are automatically saved to the database:

1. **Experiment Record** - Created automatically if doesn't exist
   - Name: Your experiment_name
   - Description: Auto-generated description

2. **Run Record** - Main training execution record
   - Training mode, status, timestamps
   - Link to experiment

3. **Configuration Record** - Your hyperparameters
   - Learning rate, epochs, batch size, weight decay, seed
   - Plus federated fields if applicable

4. **Metrics Records** - Final validation metrics
   - Loss, accuracy, precision, recall, F1-score
   - Any other metrics tracked by PyTorch Lightning

5. **Artifact Record** - Your best model
   - Path to the saved checkpoint
   - Marked as "best_model" artifact

## Retrieving Saved Results

You can query the database to retrieve saved training results:

```python
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.engine import get_session

db = get_session()

# Get a specific run
run = run_crud.get(db, run_id=123)
print(f"Run status: {run.status}")
print(f"Training mode: {run.training_mode}")

# Get all runs for an experiment
runs = run_crud.get_by_experiment(db, experiment_id=5)
for run in runs:
    print(f"Run {run.id}: {run.status}")

# Get configuration for a run
config = run.configuration
print(f"Learning rate: {config.learning_rate}")
print(f"Epochs: {config.epochs}")

# Get metrics for a run
for metric in run.metrics:
    print(f"{metric.metric_name}: {metric.metric_value}")

# Get artifacts for a run
for artifact in run.artifacts:
    print(f"{artifact.artifact_name}: {artifact.artifact_path}")

db.close()
```

## Manual Database Save (Advanced)

If you need to manually save results without full training, you can call the method directly:

```python
from federated_pneumonia_detection.src.control.dl_model.utils.data.trainer_builder import TrainerBuilder
from federated_pneumonia_detection.src.control.dl_model.utils import DatasetPreparer

# Assuming you have trainer, model, and experiment_id
trainer_builder = TrainerBuilder(
    constants=constants,
    config=config,
    checkpoint_dir="results/checkpoints",
    logs_dir="results/logs",
    logger=logger
)

# Save results
run_id = trainer_builder.save_results_to_db(
    trainer=trained_trainer,
    model=trained_model,
    experiment_id=experiment_id,
    training_mode="centralized",
    source_path="Training"
)

print(f"Saved with run ID: {run_id}")
```

## Federated Training

The same automatic save works for federated training. Just use FederatedTrainer the same way, and results are saved with training_mode="federated":

```python
from federated_pneumonia_detection.src.control.federated_learning import FederatedTrainer

trainer = FederatedTrainer(config=config, constants=constants, device=device)

results = trainer.train(
    data_df=data_df,
    image_dir=image_dir,
    experiment_name="pneumonia_federated"
)

# Results will have run_id if database save is implemented in FederatedTrainer
if 'run_id' in results:
    print(f"Federated training run saved with ID: {results['run_id']}")
```

## Troubleshooting

### Database save fails but training continues
- This is expected behavior - training completes successfully even if database save fails
- Check logs for the exact error message
- Verify database connection settings

### run_id not in results
- Database save may have failed silently (check logs)
- Database connection issues
- Verify PostgreSQL is running and accessible

### Experiment not found
- New experiment is automatically created
- Check database for the experiment record

## Database Schema

The implementation uses these existing tables:
- `experiments` - Experiment records
- `runs` - Training run records
- `run_configurations` - Hyperparameter configurations
- `run_metrics` - Training metrics
- `run_artifacts` - Model artifacts

No schema changes were needed - uses existing database design.
