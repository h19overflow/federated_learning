# Unit Tests - Quick Reference Guide

## File Locations

```
C:\Users\User\Projects\FYP2\federated_pneumonia_detection\tests\unit\control\
├── test_client.py           (26 tests)
├── test_partitioner.py      (21 tests)
├── test_data_manager.py     (26 tests)
├── test_trainer.py          (21 tests)
├── README.md                (Documentation)
├── TEST_COVERAGE.md         (Detailed coverage)
└── CHECKLIST.md             (Completion checklist)
```

## Test Execution

### Run All Tests
```bash
cd C:\Users\User\Projects\FYP2
pytest federated_pneumonia_detection/tests/unit/control/ -v
```

### Run Specific Test File
```bash
pytest federated_pneumonia_detection/tests/unit/control/test_client.py -v
pytest federated_pneumonia_detection/tests/unit/control/test_partitioner.py -v
pytest federated_pneumonia_detection/tests/unit/control/test_data_manager.py -v
pytest federated_pneumonia_detection/tests/unit/control/test_trainer.py -v
```

### Run Specific Test Class
```bash
pytest federated_pneumonia_detection/tests/unit/control/test_client.py::TestGetWeights -v
pytest federated_pneumonia_detection/tests/unit/control/test_client.py::TestFlowerClient -v
pytest federated_pneumonia_detection/tests/unit/control/test_partitioner.py::TestPartitionDataStratified -v
```

### Run Specific Test
```bash
pytest federated_pneumonia_detection/tests/unit/control/test_client.py::TestGetWeights::test_get_weights_returns_list -v
```

### Run with Minimal Output
```bash
pytest federated_pneumonia_detection/tests/unit/control/ -q
```

### Run with Coverage
```bash
pytest federated_pneumonia_detection/tests/unit/control/ --cov=federated_pneumonia_detection.src.control.federated_learning
```

### Run with Coverage Report (HTML)
```bash
pytest federated_pneumonia_detection/tests/unit/control/ --cov=federated_pneumonia_detection.src.control.federated_learning --cov-report=html
```

### Run with Detailed Failure Information
```bash
pytest federated_pneumonia_detection/tests/unit/control/ -vv --tb=long
```

### Run Tests Matching Pattern
```bash
pytest federated_pneumonia_detection/tests/unit/control/ -k "test_train" -v
pytest federated_pneumonia_detection/tests/unit/control/ -k "test_partition" -v
```

### Collect Tests (Don't Run)
```bash
pytest federated_pneumonia_detection/tests/unit/control/ --collect-only -q
```

## Test Statistics

| Module | File | Tests | Status |
|--------|------|-------|--------|
| FlowerClient | test_client.py | 26 | ✅ PASS |
| Partitioner | test_partitioner.py | 21 | ✅ PASS |
| DataManager | test_data_manager.py | 26 | ✅ PASS |
| Trainer | test_trainer.py | 21 | ✅ PASS |
| **TOTAL** | | **94** | **✅ ALL PASS** |

## Test Breakdown

### test_client.py (26 tests)
- TestGetWeights: 4 tests
- TestSetWeights: 4 tests
- TestFlowerClientInit: 4 tests
- TestFlowerClientGetParameters: 2 tests
- TestFlowerClientSetParameters: 1 test
- TestTrainFunction: 5 tests
- TestEvaluateFunction: 4 tests
- TestFlowerClientFit: 1 test
- TestFlowerClientEvaluate: 1 test

### test_partitioner.py (21 tests)
- TestPartitionDataStratified: 21 tests
  - Basic functionality
  - Data preservation
  - Reproducibility
  - Error handling
  - Edge cases

### test_data_manager.py (26 tests)
- TestSplitPartition: 9 tests
- TestLoadData: 17 tests
  - DataLoader creation
  - Configuration handling
  - Error scenarios
  - Integration tests

### test_trainer.py (21 tests)
- TestFederatedTrainerInit: 4 tests
- TestFederatedTrainerCreateModel: 4 tests
- TestFederatedTrainerGetInitialParameters: 2 tests
- TestFederatedTrainerClientFn: 2 tests
- TestFederatedTrainerCreateEvaluateFn: 3 tests
- TestFederatedTrainerTrain: 3 tests
- TestFederatedTrainerIntegration: 2 tests

## Module Reference

### client.py
Functions tested:
- `get_weights(net)` - Extract model weights
- `set_weights(net, parameters)` - Load model weights
- `train(net, trainloader, epochs, device, learning_rate, weight_decay, num_classes)` - Train model
- `evaluate(net, valloader, device, num_classes)` - Evaluate model
- `FlowerClient.__init__()`
- `FlowerClient.get_parameters()`
- `FlowerClient.set_parameters()`
- `FlowerClient.fit()`
- `FlowerClient.evaluate()`

### partitioner.py
Functions tested:
- `partition_data_stratified(df, num_clients, target_column, seed)` - Partition data across clients

### data_manager.py
Functions tested:
- `split_partition(partition_df, validation_split, target_column, seed)` - Split partition
- `load_data(partition_df, image_dir, constants, config, validation_split)` - Load data

### trainer.py
Methods tested:
- `FederatedTrainer.__init__()`
- `FederatedTrainer._create_model()`
- `FederatedTrainer._get_initial_parameters()`
- `FederatedTrainer._client_fn()`
- `FederatedTrainer._create_evaluate_fn()`
- `FederatedTrainer.train()`

## Common Issues & Solutions

### Import Errors
```bash
# Install project in editable mode
pip install -e .
```

### Missing Dependencies
```bash
# Ensure all dependencies are installed
pip install pytest torch pandas numpy scikit-learn pillow
```

### Temporary File Errors
- Tests use `tmp_path` fixture
- Ensure system temp directory is writable
- Check disk space availability

### GPU Warnings
- Tests run on CPU by default
- Pin memory warnings can be safely ignored
- GPU tests require CUDA-enabled system

## Performance Tips

### Run Faster
```bash
# Disable warnings
pytest federated_pneumonia_detection/tests/unit/control/ -q

# Run specific tests only
pytest federated_pneumonia_detection/tests/unit/control/test_client.py -q

# Parallel execution (if available)
pytest federated_pneumonia_detection/tests/unit/control/ -n auto
```

### Verbose Output
```bash
# Show print statements
pytest federated_pneumonia_detection/tests/unit/control/ -s -v

# Show local variables on failure
pytest federated_pneumonia_detection/tests/unit/control/ -l

# Show full diff on assertion failure
pytest federated_pneumonia_detection/tests/unit/control/ -vv
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run control layer tests
  run: |
    pytest federated_pneumonia_detection/tests/unit/control/ \
      -v \
      --tb=short \
      --junitxml=test-results.xml \
      --cov=federated_pneumonia_detection.src.control.federated_learning \
      --cov-report=xml
```

### Jenkins Example
```groovy
stage('Test') {
    steps {
        sh 'pytest federated_pneumonia_detection/tests/unit/control/ -v --junitxml=results.xml'
    }
    post {
        always {
            junit 'results.xml'
        }
    }
}
```

## Documentation References

- **README.md** - Quick start and overview
- **TEST_COVERAGE.md** - Detailed coverage information
- **CHECKLIST.md** - Completion verification
- **UNIT_TESTS_SUMMARY.md** - Project-level summary

## Support

For issues or questions:
1. Check the README.md in the test directory
2. Review TEST_COVERAGE.md for detailed information
3. Check CHECKLIST.md for completion status
4. See individual test files for implementation details
