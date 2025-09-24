# Test Suite Documentation

Comprehensive test suite for the Federated Pneumonia Detection System.

## Structure

```
tests/
├── unit/                           # Unit tests
│   ├── entities/                   # Entity class tests
│   │   ├── test_system_constants.py
│   │   ├── test_experiment_config.py
│   │   └── test_*.py               # Other entity tests
│   ├── control/                    # Control logic tests
│   ├── boundary/                   # Boundary interface tests
│   └── utils/                      # Utility function tests
│       ├── test_data_processing.py
│       ├── test_config_loader.py
│       └── test_*.py
├── integration/                    # Integration tests
│   ├── data_pipeline/              # Data processing integration
│   │   └── test_end_to_end_data_flow.py
│   ├── training/                   # Model training integration
│   ├── federated_learning/         # FL integration tests
│   ├── api/                        # FastAPI integration tests
│   └── dashboard/                  # Streamlit integration tests
├── fixtures/                       # Test data and fixtures
│   └── sample_data.py
├── utils/                          # Test utilities
│   └── test_helpers.py
├── conftest.py                     # Shared fixtures
├── pytest.ini                     # Pytest configuration
├── test_config.yaml                # Test-specific configuration
└── README.md                       # This file
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest -m unit                     # Unit tests only
pytest -m integration              # Integration tests only
pytest -m "unit and not slow"      # Fast unit tests only

# Run specific test files
pytest tests/unit/entities/test_system_constants.py
pytest tests/integration/data_pipeline/

# Run with coverage
pytest --cov=src --cov-report=html
```

### Phase-Based Testing

Tests are organized by development phases:

```bash
pytest -m phase1    # Phase 1: Foundation & entities
pytest -m phase2    # Phase 2: Core ML components
pytest -m phase3    # Phase 3: Centralized training
# ... etc
```

### Test Categories

- `unit`: Unit tests for individual components
- `integration`: Integration tests for component interactions
- `slow`: Tests that take more than a few seconds
- `gpu`: Tests requiring GPU hardware
- `federated`: Federated learning specific tests

## Configuration

### Test Configuration

The test suite uses `tests/test_config.yaml` with optimized settings:
- Smaller image sizes (64x64) for faster processing
- Minimal epochs and rounds for quick execution
- CPU-only execution for consistency
- Reduced logging for cleaner output

### Environment Variables

Set these for testing:
```bash
export TESTING=1
export LOG_LEVEL=WARNING
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

## Test Data

### Fixtures Available

- `sample_constants`: Basic SystemConstants for testing
- `sample_experiment_config`: ExperimentConfig for testing
- `sample_dataframe`: Standard test DataFrame
- `temp_data_structure`: Temporary file structure with images
- `pneumonia_dataset`: Realistic pneumonia dataset
- `federated_datasets`: Multiple datasets for FL testing

### Creating Test Data

```python
from tests.fixtures.sample_data import SampleDataFactory, TempDataStructure

# Create sample metadata
df = SampleDataFactory.create_sample_metadata(num_samples=100)

# Create temporary file structure
with TempDataStructure(metadata_df=df) as paths:
    # paths contains base_path, metadata_path, images_dir
    processor = DataProcessor(paths['base_path'])
```

## Writing Tests

### Unit Test Example

```python
import pytest
from src.entities.system_constants import SystemConstants

class TestSystemConstants:
    def test_default_values(self):
        constants = SystemConstants()
        assert constants.BATCH_SIZE == 128
        assert constants.IMG_SIZE == (224, 224)

    def test_custom_values(self):
        constants = SystemConstants.create_custom(batch_size=64)
        assert constants.BATCH_SIZE == 64
```

### Integration Test Example

```python
import pytest
from tests.fixtures.sample_data import TempDataStructure

class TestDataPipeline:
    def test_end_to_end_processing(self):
        with TempDataStructure() as paths:
            # Test complete data pipeline
            processor = DataProcessor(paths['base_path'])
            train_df, val_df = processor.load_and_process_data(config)

            assert len(train_df) > 0
            assert len(val_df) > 0
```

### Test Helpers

Use `tests/utils/test_helpers.py` for common operations:

```python
from tests.utils.test_helpers import TestHelpers

# Validate DataFrame structure
TestHelpers.assert_dataframe_valid(df, ['patientId', 'Target'])

# Check train/val split
TestHelpers.assert_train_val_split_valid(train_df, val_df, 0.2)

# Create mock components
mock_processor = MockComponents.create_mock_data_processor()
```

## Best Practices

### Test Organization
- One test class per source class
- Group related tests in test methods
- Use descriptive test names that explain what is being tested

### Fixtures and Mocking
- Use fixtures for common setup
- Mock external dependencies (file system, network)
- Use parametrized tests for multiple scenarios

### Assertions
- Test both success and error cases
- Verify edge cases and boundary conditions
- Use specific assertions with clear error messages

### Performance
- Keep tests fast (< 1 second each)
- Use smaller datasets for testing
- Mock expensive operations

## CI/CD Integration

The test suite is designed for CI/CD pipelines:
- All tests pass with return code 0
- Coverage reports in multiple formats
- Test results in JUnit format
- No external dependencies required

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `PYTHONPATH` includes project root
2. **File not found**: Check that test data fixtures are properly set up
3. **Slow tests**: Use `-m "not slow"` to skip time-consuming tests
4. **GPU tests failing**: Use `CUDA_VISIBLE_DEVICES=""` to force CPU

### Debug Mode

```bash
# Run with debugging
pytest -v -s --tb=long

# Stop on first failure
pytest -x

# Run specific test with full output
pytest tests/unit/entities/test_system_constants.py::TestSystemConstants::test_default_values -v -s
```

## Future Test Development

As the project progresses through phases:

1. **Phase 1** ✅: Entity and data processing tests
2. **Phase 2**: Model and training pipeline tests
3. **Phase 3**: Centralized training integration tests
4. **Phase 4**: Dashboard and UI tests
5. **Phase 5**: Federated learning tests
6. **Phase 6**: End-to-end system tests

Each phase will add new test categories and expand the integration test coverage.