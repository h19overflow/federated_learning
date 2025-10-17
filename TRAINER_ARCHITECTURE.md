# FederatedTrainer - Architecture & Design Document

## Overview

The new `FederatedTrainer` is a simplified, production-ready orchestrator for federated learning workflows. It replaces the complex monolithic `federated_trainer.py` with a clean, modular design that strictly follows SOLID principles.

## SOLID Principles Adherence

### 1. Single Responsibility Principle (SRP)

**FederatedTrainer Class: Orchestration ONLY**

Each method has exactly one responsibility:

```python
# Public interface - ONE job: orchestrate workflow
train(source_path, experiment_name) -> Dict[str, Any]

# Private methods - focused tasks
_run_simulation()              # Coordinate simulation only
_prepare_client_dataloaders()  # Create dataloaders only
_create_client_fn()            # Factory function only
_create_model()                # Model instantiation only
_create_strategy()             # Strategy creation only
_process_results()             # Result formatting only
_save_final_model()            # Model checkpoint saving only
_setup_logging()               # Logger configuration only
```

**Component Responsibilities (Not Modified):**

- `FlowerClient` (core.fed_client): Local training (fit/evaluate)
- `ClientDataManager` (data.client_data): DataLoader creation
- `partition_data_stratified` (data.partitioner): Data partitioning
- `DataSourceExtractor`: Data extraction/validation
- `DatasetPreparer`: Dataset preparation
- `ResNetWithCustomHead` (entities): Model definition

### 2. Open-Closed Principle (OCP)

**Open for extension:**
- Configuration can be changed without modifying code
- Strategy can be swapped by overriding `_create_strategy()`
- Model can be swapped by overriding `_create_model()`
- Logging can be enhanced by overriding `_setup_logging()`

**Closed for modification:**
- No hardcoded values (all configurable)
- Clear interfaces with dependencies injected
- Data flow is predictable and stable

### 3. Liskov Substitution Principle (LSP)

- All dependencies are interfaces/abstract concepts
- FlowerClient implements NumPyClient interface
- ClientDataManager provides consistent interface
- Can swap implementations without breaking orchestrator

### 4. Interface Segregation Principle (ISP)

- FederatedTrainer only depends on needed methods:
  - `DataSourceExtractor.extract_and_validate()`
  - `DatasetPreparer.prepare_dataset()`
  - `ClientDataManager.create_dataloaders_for_partition()`
  - `FlowerClient` (standard NumPyClient interface)

- No unnecessary dependencies
- No bloated interfaces

### 5. Dependency Inversion Principle (DIP)

**High-level module (FederatedTrainer) depends on abstractions:**

```python
# Not importing concrete classes, but using them via dependency injection
strategy = self._create_strategy(initial_parameters, num_clients)
# Method creates FedAvg, but trainers can override to use other strategies

client_fn = self._create_client_fn(client_dataloaders)
# Method creates FlowerClient, but can be extended to create other client types
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FederatedTrainer                             │
│                 (Orchestrator Only)                             │
└─────────┬───────────────────────────────────────────────────────┘
          │
    ┌─────┴──────────────────────────────────────────────────┐
    │                                                           │
    ├─→ train()                                                │
    │   ├─→ DataSourceExtractor.extract_and_validate()       │
    │   ├─→ DatasetPreparer.prepare_dataset()                │
    │   ├─→ partition_data_stratified()                      │
    │   └─→ _run_simulation()                                 │
    │       ├─→ ClientDataManager.create_dataloaders_for_partition()
    │       ├─→ _prepare_client_dataloaders()                 │
    │       ├─→ _create_client_fn()                           │
    │       │   └─→ FlowerClient.to_client()                  │
    │       ├─→ _create_model()                               │
    │       │   └─→ ResNetWithCustomHead()                    │
    │       ├─→ _create_strategy()                            │
    │       │   └─→ FedAvg()                                  │
    │       ├─→ start_simulation()                            │
    │       ├─→ _process_results()                            │
    │       └─→ _save_final_model()                           │
    │
    └─────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Data Extraction & Validation

```
train(source_path)
    │
    ├─ DataSourceExtractor.extract_and_validate()
    │  ├─ Detects zip or directory
    │  ├─ Extracts zip if needed (to temp dir)
    │  ├─ Finds CSV file
    │  ├─ Finds image directory
    │  └─ Returns (image_dir, csv_path)
    │
    └─ Cleanup: DataSourceExtractor.cleanup() in finally block
```

### 2. Dataset Preparation

```
DatasetPreparer.prepare_dataset()
    │
    ├─ Load metadata from CSV
    ├─ Sample data (if configured)
    ├─ Create stratified train/val split
    └─ Returns (train_df, val_df)
```

### 3. Data Partitioning

```
partition_data_stratified()
    │
    ├─ For each class label:
    │  ├─ Split samples across clients
    │  └─ Maintain class distribution
    │
    ├─ Shuffle each partition
    └─ Returns List[pd.DataFrame]
```

### 4. Dataloader Creation

```
_prepare_client_dataloaders()
    │
    └─ For each client partition:
        ├─ Split into train/val
        ├─ Apply transforms (augmentation for train, none for val)
        ├─ Create CustomImageDataset
        ├─ Create DataLoader (batch, shuffle, num_workers)
        └─ Append to client_dataloaders list
```

### 5. Client Factory & Simulation

```
_create_client_fn() - Creates closure
    │
    └─ Returns function: client_fn(context)
        │
        └─ For each simulation round:
            ├─ Get client_id from context.node_id
            ├─ Get dataloaders from closure
            ├─ Create model
            ├─ Create FlowerClient
            ├─ Convert to Flower Client
            └─ Return to Flower framework
```

### 6. Result Processing

```
_process_results()
    │
    ├─ Extract metrics from history
    ├─ Format into results dict
    └─ Return results

_save_final_model()
    │
    ├─ Get final parameters from history
    ├─ Create model
    ├─ Load parameters
    ├─ Save state_dict to checkpoint
    └─ Log path
```

## Class Structure

### FederatedTrainer

**Attributes:**
```python
checkpoint_dir: str              # Path to save checkpoints
logs_dir: str                    # Path to save logs
logger: logging.Logger           # Logger instance
constants: SystemConstants       # System configuration
config: ExperimentConfig         # Experiment configuration
data_extractor: DataSourceExtractor
dataset_preparer: DatasetPreparer
```

**Public Methods:**
```python
def __init__(config_path, checkpoint_dir, logs_dir) -> None
def train(source_path, experiment_name, csv_filename) -> Dict[str, Any]
```

**Private Methods:**
```python
def _run_simulation(...) -> Dict[str, Any]
def _prepare_client_dataloaders(...) -> List[Tuple]
def _create_client_fn(...) -> Callable
def _create_model() -> ResNetWithCustomHead
def _create_strategy(...) -> FedAvg
def _process_results(...) -> Dict[str, Any]
def _save_final_model(...) -> None
def _setup_logging() -> logging.Logger
```

## Closure Pattern for client_fn

The `_create_client_fn()` method demonstrates proper use of closures to avoid global state:

```python
def _create_client_fn(self, client_dataloaders: List[Tuple]):
    """Factory that captures dataloaders in closure."""

    def client_fn(context: Context):  # Inner function
        """Called by Flower for each client."""
        client_id = int(context.node_id)
        train_loader, val_loader = client_dataloaders[client_id]  # From closure

        model = self._create_model()  # Creates fresh model
        flower_client = FlowerClient(...)  # Fresh client
        return flower_client.to_client()

    return client_fn  # Return function, not call it
```

**Why this pattern:**
1. No global state (unlike old `client_app_factory` module)
2. Flower can serialize the function
3. Each client gets fresh model instance
4. Dataloaders are captured safely in closure
5. Thread-safe and clean

## Error Handling Strategy

### Multi-Layer Error Handling

1. **Configuration Layer**
   - Falls back to defaults if config load fails
   - Non-blocking (logs warning, continues)

2. **Data Extraction Layer**
   - Validates source path exists
   - Detects zip vs directory
   - Raises FileNotFoundError if invalid

3. **DataLoader Creation Layer**
   - Skips empty partitions (logs warning)
   - Raises if no valid dataloaders created

4. **Simulation Layer**
   - Catches all Flower exceptions
   - Logs with context
   - Re-raises with full traceback

5. **Model Saving Layer**
   - Handles missing parameters gracefully
   - Logs error if save fails (non-blocking)

### Error Messages

All error messages include:
- What went wrong
- Where it happened
- Context (file paths, client IDs, etc.)
- How to fix it (in most cases)

Example:
```python
raise ValueError(f"Client {i} out of range: expected < {len(client_dataloaders)}, got {i}")
```

## Configuration Flow

```
config_path (optional)
    │
    ├─ If provided: ConfigLoader.load_config(path)
    │  └─ Parse YAML
    │
    ├─ ConfigLoader.create_system_constants()
    │  └─ SystemConstants with paths, columns, etc.
    │
    ├─ ConfigLoader.create_experiment_config()
    │  └─ ExperimentConfig with learning_rate, num_rounds, etc.
    │
    └─ If any step fails: Use defaults (graceful degradation)
```

## Performance Considerations

### Memory Efficiency
1. Data partitioned across clients (reduces per-client memory)
2. Dataloaders use batch_size from config (default 128)
3. No unnecessary copies of full dataset

### Computation
1. Local training on each client (distributed)
2. Server aggregation only (lightweight)
3. Validation after each round (optional)

### I/O
1. Zip extraction to temp directory
2. Cleanup in finally block (guaranteed)
3. Checkpoint saving after training

## Testing Strategy

### Unit Tests

```python
def test_init_creates_directories():
    trainer = FederatedTrainer()
    assert os.path.exists(trainer.checkpoint_dir)
    assert os.path.exists(trainer.logs_dir)

def test_init_loads_config():
    trainer = FederatedTrainer()
    assert trainer.config is not None
    assert trainer.constants is not None

def test_create_client_fn_returns_callable():
    trainer = FederatedTrainer()
    mock_loader = Mock()
    dataloaders = [(mock_loader, mock_loader)]
    client_fn = trainer._create_client_fn(dataloaders)
    assert callable(client_fn)

def test_create_model_returns_model():
    trainer = FederatedTrainer()
    model = trainer._create_model()
    assert isinstance(model, ResNetWithCustomHead)
```

### Integration Tests

```python
def test_full_workflow_with_mock_data():
    trainer = FederatedTrainer()
    results = trainer.train(mock_zip_path)
    assert results['status'] == 'completed'
    assert results['num_clients'] > 0
    assert os.path.exists(results['checkpoint_dir'])
```

## Comparison with Old Implementation

| Aspect | Old federated_trainer.py | New trainer.py |
|--------|--------------------------|-----------------|
| Lines | 407 | 456 (better organized) |
| Global State | Yes (client_app_factory) | No (closure only) |
| SRP | Mixed (data, logic, orchestration) | Clear (orchestration only) |
| Type Hints | Partial | Complete |
| Error Handling | Basic | Comprehensive |
| Testability | Low (global state) | High (dependency injection) |
| Documentation | Minimal | Extensive |
| Maintainability | Low (intertwined logic) | High (clear separation) |

## Extension Points

### Override _create_model()

```python
class CustomFederatedTrainer(FederatedTrainer):
    def _create_model(self):
        # Use different architecture
        return CustomModel(self.constants, self.config)
```

### Override _create_strategy()

```python
class CustomFederatedTrainer(FederatedTrainer):
    def _create_strategy(self, initial_parameters, num_clients):
        # Use FedProx or other strategy
        return FedProx(initial_parameters=initial_parameters, ...)
```

### Override _setup_logging()

```python
class CustomFederatedTrainer(FederatedTrainer):
    def _setup_logging(self):
        # Add file handler, custom formatter
        logger = super()._setup_logging()
        file_handler = logging.FileHandler('training.log')
        logger.addHandler(file_handler)
        return logger
```

## Conclusion

The new `FederatedTrainer` achieves:
1. **Clarity**: Each method does exactly one thing
2. **Reusability**: Components can be used independently
3. **Testability**: No global state, easy to mock
4. **Maintainability**: Clear data flow and error handling
5. **Extensibility**: Easy to override and customize
6. **Production-Ready**: Comprehensive error handling and logging
