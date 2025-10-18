# Database Save Implementation Summary

## Overview
Added functionality to save training results to the database after model training completes. This includes creating run records, saving configurations, metrics, and artifacts.

## Files Modified

### 1. `federated_pneumonia_detection/src/control/dl_model/utils/data/trainer_builder.py`

**Changes:**
- Added new imports:
  - `Optional` from typing
  - `Session` from sqlalchemy.orm
  - CRUD operations: `run_crud`, `run_configuration_crud`, `run_metric_crud`, `run_artifact_crud`
  - `get_session` from engine
  - `datetime` from datetime module

- Added new method: `save_results_to_db()`
  - Saves complete training results to the database after training completes
  - Creates a Run record with experiment reference, training mode, and status
  - Saves RunConfiguration with hyperparameters (learning_rate, epochs, batch_size, etc.)
  - Saves RunMetrics with final validation metrics
  - Saves RunArtifacts with the best model checkpoint path
  - Returns the created run ID
  - Includes proper error handling with rollback on failure

**Method Signature:**
```python
def save_results_to_db(
    self,
    trainer: pl.Trainer,
    model: LitResNet,
    experiment_id: int,
    training_mode: str = "centralized",
    source_path: Optional[str] = None
) -> Optional[int]
```

### 2. `federated_pneumonia_detection/src/control/dl_model/centralized_trainer.py`

**Changes:**
- Added new imports:
  - `experiment_crud` from CRUD operations
  - `get_session` from engine

- Modified the `train()` method:
  - After results collection (Step 6), added a new step to save results to database
  - Gets or creates an experiment record using the experiment name
  - Calls `save_results_to_db()` method with the trained trainer, model, and experiment ID
  - Adds `run_id` to the returned results dictionary
  - Includes graceful error handling: if database save fails, training still completes successfully with a warning

**New Workflow:**
```
Step 6: Train model
    ↓
Collect training results
    ↓
NEW: Save results to database ← Creates Run, RunConfiguration, RunMetrics, RunArtifacts
    ↓
Return results with run_id
```

## Database Tables Populated

The implementation saves data to the following tables:

1. **runs** - Main training run record
   - experiment_id, training_mode, status, start_time, end_time, source_path

2. **run_configurations** - Hyperparameters used for training
   - learning_rate, epochs, batch_size, weight_decay, seed, etc.

3. **run_metrics** - Final validation metrics
   - metric_name, metric_value, step, dataset_type

4. **run_artifacts** - Model checkpoint paths
   - artifact_name, artifact_path, artifact_type

## Error Handling

- Database save failures don't stop training execution
- Errors are logged but training continues
- Transaction rollback on any database error
- Graceful cleanup of database sessions
- Returns run_id on success, or None if not available

## Usage Example

The database save now happens automatically during training:

```python
trainer = CentralizedTrainer(
    config_path=config_path,
    checkpoint_dir=checkpoint_dir,
    logs_dir=logs_dir
)

results = trainer.train(
    source_path=source_path,
    experiment_name=experiment_name,
    csv_filename="stage2_train_metadata.csv"
)

# results now contains 'run_id' with the database record ID
print(f"Training run saved with ID: {results['run_id']}")
```

## Notes

- Compatible with both centralized and federated training (federated config fields are optional)
- Uses existing CRUD operations for consistency
- Follows the project's database schema
- Properly handles session cleanup
- Ready for similar implementation in federated trainer
