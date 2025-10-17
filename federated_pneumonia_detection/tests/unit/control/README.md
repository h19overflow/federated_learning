# Unit Tests for Federated Learning Control Layer

Comprehensive test suite for the federated learning control layer modules.

## Overview

This directory contains unit tests for the following modules in the federated learning control layer:

- **client.py** - FlowerClient implementation for local model training
- **partitioner.py** - Data partitioning utility for stratified sampling across clients
- **data_manager.py** - Data loading and splitting for client datasets
- **trainer.py** - FederatedTrainer for orchestrating FL pipeline

## Quick Start

### Run all control layer tests:
```bash
pytest federated_pneumonia_detection/tests/unit/control/ -v
```

### Run specific test file:
```bash
pytest federated_pneumonia_detection/tests/unit/control/test_client.py -v
pytest federated_pneumonia_detection/tests/unit/control/test_partitioner.py -v
pytest federated_pneumonia_detection/tests/unit/control/test_data_manager.py -v
pytest federated_pneumonia_detection/tests/unit/control/test_trainer.py -v
```

### Run with coverage:
```bash
pytest federated_pneumonia_detection/tests/unit/control/ --cov=federated_pneumonia_detection.src.control.federated_learning --cov-report=html
```

## Test Files

### test_client.py (26 tests)
Tests for the FlowerClient class and helper functions:
- `get_weights()` - Weight extraction (4 tests)
- `set_weights()` - Weight loading (4 tests)
- `train()` - Model training (5 tests)
- `evaluate()` - Model evaluation (4 tests)
- `FlowerClient` - Client class (9 tests)

### test_partitioner.py (21 tests)
Tests for data partitioning:
- `partition_data_stratified()` - Stratified partitioning across clients
- Covers edge cases: empty data, imbalanced classes, reproducibility

### test_data_manager.py (26 tests)
Tests for data loading and splitting:
- `split_partition()` - Train/validation splitting (7 tests)
- `load_data()` - DataLoader creation (19 tests)
- Covers: batching, shuffling, augmentation, error handling

### test_trainer.py (21 tests)
Tests for federated training orchestration:
- `FederatedTrainer` - Trainer class initialization and methods
- Model creation, parameter initialization, error handling

## Test Statistics

| Test File | # Tests | Status |
|-----------|---------|--------|
| test_client.py | 26 | ✅ PASS |
| test_partitioner.py | 21 | ✅ PASS |
| test_data_manager.py | 26 | ✅ PASS |
| test_trainer.py | 21 | ✅ PASS |
| **TOTAL** | **94** | **✅ ALL PASS** |

## Key Features

✅ **Comprehensive Coverage**
- Input validation and error handling
- Edge cases and boundary conditions
- Multi-class and binary classification support
- Device placement (CPU/GPU)

✅ **Reproducibility**
- Seed-based deterministic behavior
- Multiple test scenarios per functionality
- Mock objects for dependencies

✅ **Well-Organized**
- Tests grouped by class/function
- Clear naming conventions
- Fixtures for common setup

✅ **Error Scenarios**
- None/invalid inputs
- Missing required columns
- Empty datasets
- Invalid configurations
- Missing files/directories

## Test Examples

### Testing weight extraction and loading
```python
def test_get_weights_returns_list():
    net = SimpleTestNet()
    weights = get_weights(net)
    assert isinstance(weights, list)
    assert all(isinstance(w, np.ndarray) for w in weights)

def test_set_weights_updates_model():
    net = SimpleTestNet()
    original_weights = get_weights(net)
    modified_weights = [np.random.randn(*w.shape) for w in original_weights]
    set_weights(net, modified_weights)
    new_weights = get_weights(net)
    for new, modified in zip(new_weights, modified_weights):
        assert np.allclose(new, modified)
```

### Testing data partitioning
```python
def test_partition_preserves_total_samples():
    result = partition_data_stratified(df, num_clients=3, ...)
    total = sum(len(df) for df in result)
    assert total == len(original_df)

def test_partition_reproducible_with_seed():
    result1 = partition_data_stratified(df, seed=42, ...)
    result2 = partition_data_stratified(df, seed=42, ...)
    # Results should be identical
```

## Dependencies

- pytest>=8.4.2
- torch>=2.8.0
- pandas>=2.3.2
- numpy>=1.x
- scikit-learn>=1.7.2

## Implementation Notes

### Mock Patterns
- DataLoaders are mocked using `MockDataLoader` class to avoid actual file I/O
- Flower simulation is mocked to test trainer logic independently
- System constants and configuration objects are used directly

### Test Network
A `SimpleTestNet` class is provided for testing:
```python
class SimpleTestNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, num_classes)
```

### Fixtures
Common fixtures provided:
- `sample_partition()` - DataFrame with test data
- `balanced_dataframe()` - Balanced class distribution
- `mock_setup()` - Complete test environment with temporary files

## Running Tests in CI/CD

```yaml
# Example GitHub Actions workflow
- name: Run control layer tests
  run: |
    pytest federated_pneumonia_detection/tests/unit/control/ \
      -v \
      --tb=short \
      --junitxml=test-results.xml
```

## Troubleshooting

### Import Errors
Ensure the project is installed in editable mode:
```bash
pip install -e .
```

### GPU-related Warnings
Tests run on CPU by default. GPU warnings about pin_memory can be safely ignored.

### Temporary File Errors
Tests use `tmp_path` fixture for temporary files. If tests fail due to path issues, ensure the system temp directory is writable.

## Contributing

When adding new tests:
1. Follow the existing naming convention: `test_<functionality>`
2. Group related tests in classes: `Test<ComponentName>`
3. Use descriptive docstrings
4. Add appropriate fixtures for reusable setup
5. Test both success and error cases
6. Run all tests before submitting: `pytest federated_pneumonia_detection/tests/unit/control/ -v`

## Documentation

For detailed coverage information, see [TEST_COVERAGE.md](TEST_COVERAGE.md)
