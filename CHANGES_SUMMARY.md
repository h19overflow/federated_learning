# MetricsCollectorCallback - Foreign Key Relationship Fix

## Problem
The `MetricsCollectorCallback` in `metrics_collector.py` was attempting to persist metrics to the database without ensuring that the `Run` record existed first. Since `RunMetric` has a foreign key constraint to `Run` (run_id references runs.id), this caused FK relationship violations when only pushing metrics.

## Root Cause
- `RunMetric` table has `run_id` as a foreign key to `Run.id`
- `Run` table requires `experiment_id` as a foreign key to `Experiment.id`
- Metrics were being persisted without creating the parent `Run` record first

## Solution
Modified `federated_pneumonia_detection/src/control/dl_model/utils/model/metrics_collector.py`:

### Changes Made

#### 1. Added Import
```python
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
```

#### 2. Updated Constructor Signature
Added two new parameters to `__init__`:
- `experiment_id: Optional[int] = None` - Required for creating a run if it doesn't exist
- `training_mode: str = "centralized"` - The training mode for the run

**New signature:**
```python
def __init__(
    self,
    save_dir: str,
    experiment_name: str = "experiment",
    run_id: Optional[int] = None,
    experiment_id: Optional[int] = None,
    training_mode: str = "centralized",
    enable_db_persistence: bool = True
)
```

#### 3. Added `_ensure_run_exists()` Method
New method that:
- Verifies if a run already exists when `run_id` is provided
- Creates a new `Run` record if `run_id` is None
- Requires `experiment_id` to create a run (raises `ValueError` if missing)
- Sets the run status to 'in_progress' initially
- Returns the `run_id` for use in metric persistence

```python
def _ensure_run_exists(self, db: Session) -> int:
    """
    Ensure run exists in database. Create it if necessary.
    
    Returns:
        run_id: The ID of the run
    """
    if self.run_id is not None:
        existing_run = run_crud.get(db, self.run_id)
        if existing_run:
            return self.run_id
    
    if self.experiment_id is None:
        raise ValueError(
            "experiment_id is required to create a run. "
            "Please provide experiment_id when initializing MetricsCollectorCallback"
        )
    
    run_data = {
        'experiment_id': self.experiment_id,
        'training_mode': self.training_mode,
        'status': 'in_progress',
        'start_time': self.training_start_time or datetime.now(),
    }
    
    new_run = run_crud.create(db, **run_data)
    db.flush()
    self.run_id = new_run.id
    
    return self.run_id
```

#### 4. Updated `persist_to_database()` Method
- Removed the check that skipped persistence when `run_id` is None
- Calls `_ensure_run_exists(db)` to guarantee run exists before persisting metrics
- Uses the returned `run_id` for metric persistence

## Usage

### Before
```python
# Would fail with FK constraint error if run doesn't exist
callback = MetricsCollectorCallback(
    save_dir="./metrics",
    experiment_name="exp1",
    enable_db_persistence=True
)
```

### After - Create Run Automatically
```python
# Run will be created automatically if it doesn't exist
callback = MetricsCollectorCallback(
    save_dir="./metrics",
    experiment_name="exp1",
    experiment_id=1,  # Required for auto-creation
    training_mode="federated",
    enable_db_persistence=True
)
```

### After - Use Existing Run
```python
# Use existing run by providing run_id
callback = MetricsCollectorCallback(
    save_dir="./metrics",
    experiment_name="exp1",
    run_id=42,  # Existing run ID
    enable_db_persistence=True
)
```

## Database Schema Context

### Run Table (Parent)
```
- id (PK)
- experiment_id (FK -> experiments.id) [REQUIRED]
- training_mode
- status
- start_time
- end_time
```

### RunMetric Table (Child)
```
- id (PK)
- run_id (FK -> runs.id) [REQUIRED]
- metric_name
- metric_value
- step
- dataset_type
```

## Benefits
1. **FK Constraint Satisfaction**: Run is created before metrics are persisted
2. **Automatic Run Creation**: No manual run creation needed if `experiment_id` is provided
3. **Backward Compatible**: Existing code using `run_id` continues to work
4. **Clear Error Messages**: Explicit error if `experiment_id` is missing when needed
5. **Consistent State**: Run status set to 'in_progress' during creation
