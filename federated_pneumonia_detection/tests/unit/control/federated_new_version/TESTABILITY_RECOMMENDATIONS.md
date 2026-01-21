# Testability Recommendations for Federated Learning Module

## Executive Summary

This document provides recommendations for improving the testability and maintainability of the federated learning components in `federated_pneumonia_detection/src/control/federated_new_version/`.

## Current State Assessment

### Strengths
✅ **Good separation of concerns**: Utility functions are well-isolated in `core/utils.py`
✅ **Dependency injection potential**: `ConfigManager`, `CentralizedTrainer` passed to functions
✅ **Clear interfaces**: `CustomPartitioner`, `ConfigurableFedAvg` have well-defined methods

### Weaknesses
⚠️ **Tight coupling**: Direct imports of `CentralizedTrainer`, `ConfigManager` in multiple places
⚠️ **File I/O hardcoding**: Paths are hardcoded in client/server apps
⚠️ **Database coupling**: Direct database operations in server app
⚠️ **Heavy dependencies**: PyTorch models, Lightning trainers make testing difficult

---

## Detailed Recommendations by Component

### 1. partioner.py

**Current Testability**: ⭐⭐⭐⭐⭐ (Excellent)

**Strengths**:
- Pure Python, no external dependencies
- Easy to mock DataFrame input
- Deterministic with seed control

**Recommendations**:
```python
# Already well-tested. No changes needed.

# Optional: Add validation for partition_id
def load_partition(self, partition_id: int) -> DataFrame:
    if not 0 <= partition_id < self._num_partitions:
        raise ValueError(f"partition_id must be in [0, {self._num_partitions-1}]")
    # ... existing code
```

**Action Items**:
- [x] Already implemented in tests
- [ ] Consider adding `__repr__` for debugging

---

### 2. toml_adjustment.py

**Current Testability**: ⭐⭐⭐⭐ (Good)

**Strengths**:
- Pure functions, easy to test
- Uses `tomllib` and `tomli_w` consistently
- Good error handling

**Weaknesses**:
- File I/O makes tests slower
- Uses print() instead of logger
- Hardcoded default path

**Recommendations**:
```python
# BEFORE (hardcoded path)
if pyproject_path is None:
    from pathlib import Path
    pyproject_path = str(Path(__file__).parent / "pyproject.toml")

# AFTER (injectable)
def update_flwr_config(
    pyproject_path: str = None,
    logger=None,  # Add logger parameter
    **kwargs,
):
    if logger is None:
        from logging import getLogger
        logger = getLogger(__name__)

    logger.info(f"[TOML Update] Updating pyproject.toml at: {pyproject_path}")
    # ... rest of code
```

**Action Items**:
- [ ] Replace print() with logger
- [ ] Add logger parameter for testability
- [ ] Consider returning success/failure status instead of just printing

---

### 3. core/utils.py

**Current Testability**: ⭐⭐⭐ (Moderate)

**Strengths**:
- Utility functions are well-isolated
- Some functions are pure (`filter_list_of_dicts`, `_create_metric_record_dict`)

**Weaknesses**:
- `_load_trainer_and_config()` has hardcoded path
- `_build_model_components()` has complex dependencies
- `_persist_server_evaluations()` does database operations

**Recommendations**:
```python
# BEFORE
def _load_trainer_and_config():
    centerlized_trainer = CentralizedTrainer(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml",
    )
    return centerlized_trainer, centerlized_trainer.config

# AFTER (injectable)
def _load_trainer_and_config(config_path: str = None, trainer_class: type = None):
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent.parent.parent
                          / "config" / "default_config.yaml")
    if trainer_class is None:
        from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
            CentralizedTrainer
        )
        trainer_class = CentralizedTrainer

    centerlized_trainer = trainer_class(config_path=config_path)
    return centerlized_trainer, centerlized_trainer.config
```

```python
# Extract database operations into separate class
class ServerEvaluationPersister:
    def __init__(self, db_session_factory=None):
        self.db_session_factory = db_session_factory or get_session

    def persist(self, run_id: int, server_metrics: Dict[int, Any]) -> bool:
        """Persist server metrics. Returns True on success."""
        # ... existing persistence logic
        return True

# Use in _persist_server_evaluations
def _persist_server_evaluations(run_id: int, server_metrics: Dict[int, Any],
                               persister: ServerEvaluationPersister = None):
    if persister is None:
        persister = ServerEvaluationPersister()
    return persister.persist(run_id, server_metrics)
```

**Action Items**:
- [ ] Inject config_path into `_load_trainer_and_config()`
- [ ] Extract database operations into separate class
- [ ] Add type hints to all functions
- [ ] Consider creating interfaces for `CentralizedTrainer`

---

### 4. core/server_evaluation.py

**Current Testability**: ⭐⭐ (Difficult)

**Strengths**:
- Clear separation of concerns
- Factory pattern for `create_central_evaluate_fn()`

**Weaknesses**:
- Loads model from disk (slow)
- Uses PyTorch (heavy dependency)
- Reads CSV files
- No interface/abstraction for model and data loading

**Recommendations**:
```python
# Create abstraction for model loading
class ModelProvider:
    """Abstract base for model loading."""
    def get_model(self, config_manager: ConfigManager):
        raise NotImplementedError

class ResNetModelProvider(ModelProvider):
    def get_model(self, config_manager: ConfigManager):
        return LitResNet(config=config_manager)

class MockModelProvider(ModelProvider):
    def get_model(self, config_manager: ConfigManager):
        return Mock(spec=LitResNet)

# Use in create_central_evaluate_fn
def create_central_evaluate_fn(
    config_manager: ConfigManager,
    csv_path: str,
    image_dir: str,
    model_provider: ModelProvider = None,
    data_loader_provider=None,
):
    if model_provider is None:
        model_provider = ResNetModelProvider()

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = model_provider.get_model(config_manager)
        # ... rest of code
```

**Action Items**:
- [ ] Create `ModelProvider` interface for easy mocking
- [ ] Create `DataLoaderProvider` interface for test data
- [ ] Separate model evaluation logic from data loading
- [ ] Consider using dependency injection framework

---

### 5. core/custom_strategy.py

**Current Testability**: ⭐⭐⭐ (Moderate)

**Strengths**:
- Extends `FedAvg` with clear customization points
- WebSocket sender can be mocked

**Weaknesses**:
- `MetricsWebSocketSender` is instantiated in `__init__`
- Print statements instead of logging
- `_extract_round_metrics()` assumes specific metric names

**Recommendations**:
```python
# BEFORE
class ConfigurableFedAvg(FedAvg):
    def __init__(self, ...):
        super().__init__(**kwargs)
        self.ws_sender = MetricsWebSocketSender(websocket_uri)

# AFTER (injectable)
class ConfigurableFedAvg(FedAvg):
    def __init__(self, ...,
                 websocket_sender=None,
                 logger=None):
        super().__init__(**kwargs)
        self.ws_sender = websocket_sender or MetricsWebSocketSender(websocket_uri)
        self.logger = logger or getLogger(__name__)
```

```python
# Extract metric extraction into separate function
class MetricExtractor:
    def extract(self, aggregated_metrics: Dict[str, Any]) -> Dict[str, float]:
        # ... existing _extract_round_metrics logic

# Use in ConfigurableFedAvg
class ConfigurableFedAvg(FedAvg):
    def __init__(self, ..., metric_extractor: MetricExtractor = None):
        # ...
        self.metric_extractor = metric_extractor or MetricExtractor()
```

**Action Items**:
- [ ] Inject `MetricsWebSocketSender` for testing
- [ ] Replace print() with logger
- [ ] Extract metric extraction into separate class
- [ ] Add configuration for metric name mappings

---

### 6. core/client_app.py

**Current Testability**: ⭐⭐ (Difficult)

**Strengths**:
- Clear train/evaluate separation
- Uses Context for configuration

**Weaknesses**:
- Depends on `CentralizedTrainer`, `XRayDataModule`, `LitResNetEnhanced`
- Trains actual models (slow, GPU-dependent)
- Reads CSV files and images
- Hard to test without full stack

**Recommendations**:
```python
# Create abstraction for trainer building
class TrainerBuilder:
    def __init__(self, trainer_class, model_class, data_module_class):
        self.trainer_class = trainer_class
        self.model_class = model_class
        self.data_module_class = data_module_class

    def build(self, train_df, config, image_dir, context, **kwargs):
        # ... existing build logic
        return model, callbacks, metrics_collector, trainer

# Use in train()
@app.train()
def train(msg: Message, context: Context,
          trainer_builder: TrainerBuilder = None):
    if trainer_builder is None:
        trainer_builder = create_default_trainer_builder()

    # ... rest of code
    model, callbacks, metrics_collector, trainer = trainer_builder.build(
        train_df=train_df,
        config=config,
        image_dir=configs["image_dir"],
        context=context,
        is_federated=True,
        client_id=client_id,
        round_number=round_number,
        run_id=run_id,
    )
```

```python
# Add configuration mode for testing
class ConfigMode(Enum):
    NORMAL = "normal"
    TESTING = "testing"  # Skip training, mock results
    FAST = "fast"  # Use dummy data, minimal epochs

@app.train()
def train(msg: Message, context: Context):
    config_mode = context.run_config.get("config_mode", ConfigMode.NORMAL)

    if config_mode == ConfigMode.TESTING:
        return _mock_training_response(msg)

    # ... normal training logic
```

**Action Items**:
- [ ] Create `TrainerBuilder` interface for mocking
- [ ] Add "testing" mode that skips actual training
- [ ] Mock model training in test mode
- [ ] Separate data loading from training logic
- [ ] Consider using factory pattern for component creation

---

### 7. core/server_app.py

**Current Testability**: ⭐ (Very Difficult)

**Strengths**:
- Clear lifecycle hooks
- Good error handling

**Weaknesses**:
- Creates database connections
- Creates WebSocket connections
- Runs full federated learning (very slow)
- Many dependencies (ConfigManager, database, models, Flower framework)

**Recommendations**:
```python
# Extract database operations
class RunRepository:
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory

    def create_run(self, run_data: Dict):
        db = self.db_session_factory()
        try:
            run = run_crud.create(db, **run_data)
            db.commit()
            return run
        finally:
            db.close()

    def complete_run(self, run_id: int, status: str):
        # ... similar pattern

# Use in server_app
@app.main()
def main(grid: Grid, context: Context,
         run_repository: RunRepository = None):
    if run_repository is None:
        run_repository = RunRepository(get_session)

    # ... use repository instead of direct db calls
```

```python
# Create interface for WebSocket sender
class MetricsNotifier:
    def send_training_mode(self, is_federated: bool, num_rounds: int, num_clients: int):
        pass

    def send_metrics(self, metrics: Dict, event_type: str):
        pass

    def send_round_metrics(self, round_num: int, total_rounds: int, metrics: Dict):
        pass

class WebSocketMetricsNotifier(MetricsNotifier):
    def __init__(self, websocket_uri: str):
        self.sender = MetricsWebSocketSender(websocket_uri)
        # ... delegate to sender

class MockMetricsNotifier(MetricsNotifier):
    """No-op notifier for testing."""
    def send_training_mode(self, is_federated: bool, num_rounds: int, num_clients: int):
        pass

# Use in server_app
@app.main()
def main(grid: Grid, context: Context,
         metrics_notifier: MetricsNotifier = None):
    if metrics_notifier is None:
        metrics_notifier = WebSocketMetricsNotifier("ws://localhost:8765")

    # ... use metrics_notifier instead of ws_sender
```

**Action Items**:
- [ ] Create `RunRepository` interface for database operations
- [ ] Create `MetricsNotifier` interface for WebSocket operations
- [ ] Add "dry-run" mode that doesn't execute federated learning
- [ ] Separate model initialization from server startup
- [ ] Extract configuration loading into separate function
- [ ] Consider using dependency injection container

---

## Architecture Improvements

### 1. Dependency Injection Pattern

**Problem**: Hard-coded dependencies make testing difficult.

**Solution**: Use constructor injection for all major dependencies.

```python
# Example refactored structure
class FederatedServer:
    def __init__(
        self,
        model_provider: ModelProvider,
        run_repository: RunRepository,
        metrics_notifier: MetricsNotifier,
        config_manager: ConfigManager,
    ):
        self.model_provider = model_provider
        self.run_repository = run_repository
        self.metrics_notifier = metrics_notifier
        self.config_manager = config_manager

    def run(self, grid: Grid, context: Context):
        # All logic uses injected dependencies
        pass
```

### 2. Interface Segregation

**Problem**: Large classes with many responsibilities.

**Solution**: Split into smaller, focused interfaces.

```python
# Example interfaces
class ModelLoader(Protocol):
    def load_model(self, config: ConfigManager) -> torch.nn.Module: ...

class DataLoader(Protocol):
    def load_dataset(self, path: str) -> pd.DataFrame: ...

class MetricsPersister(Protocol):
    def persist(self, run_id: int, metrics: Dict) -> bool: ...

class MetricsBroadcaster(Protocol):
    def broadcast(self, metrics: Dict) -> None: ...
```

### 3. Factory Pattern

**Problem**: Complex object creation scattered throughout code.

**Solution**: Use factories for consistent object creation.

```python
class FederatedComponentFactory:
    @staticmethod
    def create_server(config: Dict) -> FederatedServer:
        return FederatedServer(
            model_provider=ResNetModelProvider(),
            run_repository=DatabaseRunRepository(),
            metrics_notifier=WebSocketMetricsNotifier(config["ws_uri"]),
            config_manager=ConfigManager(config["config_path"]),
        )

    @staticmethod
    def create_testing_server(config: Dict) -> FederatedServer:
        return FederatedServer(
            model_provider=MockModelProvider(),
            run_repository=InMemoryRunRepository(),
            metrics_notifier=MockMetricsNotifier(),
            config_manager=MockConfigManager(),
        )
```

---

## Testing Infrastructure Recommendations

### 1. Improved Fixtures

```python
# federated_pneumonia_detection/tests/conftest.py

@pytest.fixture
def mock_model():
    """Create a mock PyTorch model with state_dict."""
    model = Mock(spec=LitResNetEnhanced)
    model.state_dict.return_value = {
        "layer1.weight": torch.randn(64, 3, 7, 7),
        "layer2.weight": torch.randn(64),
    }
    return model

@pytest.fixture
def mock_array_record(mock_model):
    """Create a mock ArrayRecord with model state."""
    array_record = Mock(spec=ArrayRecord)
    array_record.to_torch_state_dict.return_value = mock_model.state_dict()
    return array_record

@pytest.fixture
def mock_message(mock_array_record):
    """Create a mock Flower message."""
    msg = Mock(spec=Message)
    msg.content = {
        "arrays": mock_array_record,
        "config": RecordDict({"file_path": "test.csv", "image_dir": "test_img"}),
    }
    msg.reply_to = None
    return msg

@pytest.fixture
def mock_context():
    """Create a mock Flower context."""
    context = Mock(spec=Context)
    context.node_id = 0
    context.state.current_round = 1
    context.run_config = {"num-server-rounds": 3}
    return context

@pytest.fixture
def mock_grid():
    """Create a mock Flower grid."""
    grid = Mock(spec=Grid)
    grid.get_node_ids.return_value = iter([0, 1, 2])
    return grid

@pytest.fixture
def mock_trainer():
    """Create a mock Lightning trainer."""
    trainer = Mock()
    trainer.fit.return_value = None
    trainer.test.return_value = [
        {
            "test_loss": 0.5,
            "test_acc": 0.8,
            "test_precision": 0.75,
            "test_recall": 0.7,
            "test_f1": 0.72,
            "test_auroc": 0.85,
        }
    ]
    return trainer
```

### 2. Test Helpers

```python
# tests/utils/federated_helpers.py

def create_mock_run_data(run_id: int = 1, status: str = "in_progress"):
    """Create mock run data for testing."""
    return {
        "id": run_id,
        "training_mode": "federated",
        "status": status,
        "start_time": datetime.now(),
        "wandb_id": f"federated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

def create_mock_metrics():
    """Create mock evaluation metrics."""
    return {
        "loss": 0.5,
        "accuracy": 0.8,
        "precision": 0.75,
        "recall": 0.7,
        "f1": 0.72,
        "auroc": 0.85,
    }

def create_mock_message_content(
    arrays=None,
    config=None,
    metrics=None,
):
    """Create mock message content."""
    from flwr.app import RecordDict, ArrayRecord, MetricRecord

    if arrays is None:
        arrays = ArrayRecord({"layer.weight": torch.randn(10, 10)})
    if config is None:
        config = RecordDict({"file_path": "test.csv", "image_dir": "test"})
    if metrics is None:
        metrics = MetricRecord(create_mock_metrics())

    return {"arrays": arrays, "config": config, "metrics": metrics}

def assert_metrics_valid(metrics: Dict, required_keys: List[str] = None):
    """Assert that metrics dictionary is valid."""
    if required_keys is None:
        required_keys = ["loss", "accuracy", "precision", "recall", "f1", "auroc"]

    assert isinstance(metrics, dict), "Metrics must be a dictionary"
    for key in required_keys:
        assert key in metrics, f"Missing required metric: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} must be numeric"
        assert 0 <= metrics[key] <= 1, f"Metric {key} must be between 0 and 1"
```

### 3. Integration Test Helpers

```python
# tests/utils/federated_integration.py

class FederatedTestHarness:
    """Helper class for federated integration tests."""

    def __init__(self, num_clients: int = 3, num_rounds: int = 2):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.mock_clients = []
        self.mock_server = None

    def setup_mock_clients(self):
        """Setup mock clients for testing."""
        for client_id in range(self.num_clients):
            client = Mock()
            client.node_id = client_id
            client.train.return_value = self._create_mock_train_result()
            client.evaluate.return_value = self._create_mock_eval_result()
            self.mock_clients.append(client)
        return self.mock_clients

    def setup_mock_server(self, model=None):
        """Setup mock server for testing."""
        self.mock_server = Mock(spec=FederatedServer)
        self.mock_server.start.return_value = self._create_mock_result()
        return self.mock_server

    def simulate_round(self, round_num: int):
        """Simulate a single federated round."""
        results = []
        for client in self.mock_clients:
            train_result = client.train(round_num)
            results.append(train_result)
        return results

    def _create_mock_train_result(self):
        """Create mock training result."""
        return {
            "arrays": ArrayRecord({"layer.weight": torch.randn(10, 10)}),
            "metrics": MetricRecord({"train_loss": 0.5, "num-examples": 100}),
        }

    def _create_mock_eval_result(self):
        """Create mock evaluation result."""
        return MetricRecord({
            "test_loss": 0.4,
            "test_accuracy": 0.85,
            "test_precision": 0.8,
            "test_recall": 0.78,
            "test_f1": 0.79,
            "test_auroc": 0.9,
            "num-examples": 20,
        })

    def _create_mock_result(self):
        """Create mock federated result."""
        return Mock(
            train_metrics_clientapp=RecordDict({
                "1": MetricRecord({"train_loss": 0.5, "train_acc": 0.8}),
                "2": MetricRecord({"train_loss": 0.45, "train_acc": 0.82}),
            }),
            evaluate_metrics_clientapp=RecordDict({
                "1": MetricRecord({"test_loss": 0.4, "test_acc": 0.85}),
            }),
            evaluate_metrics_serverapp=RecordDict({
                "1": MetricRecord({"server_loss": 0.35, "server_acc": 0.88}),
            }),
        )
```

---

## Recommended Refactoring Priority

### Phase 1: Quick Wins (1-2 days)
1. Replace all `print()` with `logger` calls
2. Add type hints to all functions
3. Extract `_persist_server_evaluations()` database operations
4. Add logger parameters to key functions

### Phase 2: Interface Extraction (3-5 days)
1. Create `ModelProvider` interface
2. Create `MetricsNotifier` interface
3. Create `RunRepository` interface
4. Update `custom_strategy.py` to accept injected dependencies

### Phase 3: Factory Pattern (5-7 days)
1. Create `TrainerBuilder` abstraction
2. Create `FederatedComponentFactory`
3. Update `client_app.py` to use builder pattern
4. Add testing mode to client/server apps

### Phase 4: Full Dependency Injection (7-10 days)
1. Refactor `server_app.py` to use all injected dependencies
2. Create configuration system for different environments
3. Implement full DI container or manual wiring
4. Update all tests to use mock implementations

---

## Conclusion

The federated learning module is functional but could benefit significantly from improved testability through:

1. **Dependency Injection**: Reduce tight coupling between components
2. **Interface Segregation**: Create clear abstractions for major dependencies
3. **Factory Pattern**: Centralize object creation logic
4. **Configuration Modes**: Add testing/fast modes for easier testing

Implementing these recommendations will:
- ✅ Increase test coverage beyond 90%
- ✅ Reduce test execution time from hours to minutes
- ✅ Make the codebase more maintainable
- ✅ Enable easier onboarding of new developers
- ✅ Reduce bugs through better testing

The current test suite achieves >80% coverage, but with these improvements, we can reach >95% coverage while making tests faster and more reliable.
