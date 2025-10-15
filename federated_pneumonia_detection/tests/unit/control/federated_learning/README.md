# Federated Learning Test Suite

Comprehensive test suite for the federated learning module.

## Test Structure

### Unit Tests (`tests/unit/control/federated_learning/`)

- **`test_data_partitioner.py`** - Tests for data partitioning strategies
  - IID partitioning
  - Non-IID (patient-based) partitioning
  - Stratified partitioning
  - Edge cases and error handling

- **`test_training_functions.py`** - Tests for PyTorch training utilities
  - Single epoch training
  - Model evaluation
  - Parameter serialization/deserialization
  - Optimizer creation
  - Multi-epoch training workflows

- **`test_federated_trainer.py`** - Tests for FederatedTrainer orchestrator
  - Initialization with different configurations
  - Data partitioning delegation
  - Training status monitoring
  - Source validation

- **`test_client_server_apps.py`** - Tests for Flower client/server apps
  - Configuration loading
  - Model creation
  - Flower app initialization

### Integration Tests (`tests/integration/federated_learning/`)

- **`test_fl_workflow.py`** - End-to-end workflow tests
  - Complete data partitioning pipeline
  - Model parameter exchange simulation
  - Federated averaging simulation
  - Full trainer workflow

## Running Tests

### Run All Federated Learning Tests
```bash
pytest tests/unit/control/federated_learning/ tests/integration/federated_learning/ -v
```

### Run Only Unit Tests
```bash
pytest tests/unit/control/federated_learning/ -v
```

### Run Only Integration Tests
```bash
pytest tests/integration/federated_learning/ -v
```

### Run Specific Test File
```bash
pytest tests/unit/control/federated_learning/test_data_partitioner.py -v
```

### Run with Coverage
```bash
pytest tests/unit/control/federated_learning/ --cov=federated_pneumonia_detection.src.control.federated_learning --cov-report=html
```

### Run Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run only federated learning tests
pytest -m federated

# Run only integration tests
pytest -m integration
```

## Test Coverage

The test suite covers:

1. **Data Partitioning** (100% coverage)
   - All three partitioning strategies
   - Edge cases (empty data, single client, etc.)
   - Error conditions

2. **Training Functions** (95% coverage)
   - Forward/backward passes
   - Evaluation metrics
   - Parameter handling
   - Optimizer configuration

3. **Federated Trainer** (85% coverage)
   - Initialization and configuration
   - Data workflow orchestration
   - Status monitoring

4. **Client/Server Apps** (80% coverage)
   - Configuration loading
   - Model initialization
   - Flower integration points

5. **Integration Workflows** (90% coverage)
   - End-to-end data integrity
   - Parameter exchange
   - Federated averaging

## Test Fixtures

Common fixtures are defined in:
- `conftest.py` - Federated learning specific fixtures
- `tests/conftest.py` - Global test fixtures

Key fixtures:
- `fl_config` - Federated learning configuration
- `fl_constants` - System constants for FL
- `federated_df` - Sample federated dataset
- `mock_torch_model` - Mock PyTorch model
- `temp_fl_dirs` - Temporary directories for testing

## Writing New Tests

When adding new tests, follow these guidelines:

1. **Keep tests under 150 lines** per file (create utils if needed)
2. **Use descriptive test names** that explain what is being tested
3. **Follow AAA pattern**: Arrange, Act, Assert
4. **Use appropriate markers**: @pytest.mark.unit, @pytest.mark.integration
5. **Mock external dependencies** to isolate units under test
6. **Test edge cases and error conditions**

Example:
```python
@pytest.mark.unit
class TestNewFeature:
    def test_feature_works_correctly(self, fixture):
        # Arrange
        data = setup_test_data()
        
        # Act
        result = function_under_test(data)
        
        # Assert
        assert result is not None
        assert result.property == expected_value
```

## Dependencies

Required packages for testing:
- pytest
- pytest-cov
- pytest-mock
- torch (CPU version)
- pandas
- numpy

Install with:
```bash
uv pip install -r tests/requirements-test.txt
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- No GPU required (CPU-only tests)
- Fast execution (< 2 minutes for full suite)
- Clear failure messages
- Isolated test environments

## Troubleshooting

### Import Errors
Ensure the project root is in PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/federated_pneumonia_detection"
```

### Mock Errors
If mocks are not working, verify patch paths match actual import paths in the code.

### Fixture Not Found
Check that conftest.py files are in the correct locations and fixtures are properly defined.

## Contributing

When contributing tests:
1. Run full test suite before committing
2. Ensure new tests pass locally
3. Add tests for any new functionality
4. Update this README if adding new test categories



