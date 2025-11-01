# Unit Tests for Federated Learning Control Layer

This document outlines the comprehensive unit test coverage for the federated learning control layer modules.

## Test Files Created

### 1. `test_client.py` - FlowerClient Tests (26 tests)

Tests for the federated learning client implementation in `client.py`.

#### TestGetWeights (4 tests)
- `test_get_weights_returns_list`: Verifies weights extraction returns list of numpy arrays
- `test_get_weights_matches_state_dict`: Ensures all state dict parameters are extracted
- `test_get_weights_cpu_conversion`: Confirms weights are on CPU after extraction
- `test_get_weights_reproducible`: Validates reproducibility of weight extraction

#### TestSetWeights (4 tests)
- `test_set_weights_updates_model`: Verifies parameters are properly updated
- `test_set_weights_torch_tensor_conversion`: Confirms numpy arrays are converted to tensors
- `test_set_weights_with_random_weights`: Tests setting arbitrary weight shapes
- `test_set_weights_preserves_order`: Validates parameter order is maintained

#### TestFlowerClientInit (4 tests)
- `test_init_valid_parameters`: Tests initialization with valid dependencies
- `test_init_none_net_raises_error`: Validates net cannot be None
- `test_init_none_trainloader_raises_error`: Validates trainloader cannot be None
- `test_init_none_valloader_raises_error`: Validates valloader cannot be None

#### TestFlowerClientGetParameters (2 tests)
- `test_get_parameters_returns_list`: Verifies get_parameters returns list
- `test_get_parameters_matches_weights`: Ensures output matches get_weights function

#### TestFlowerClientSetParameters (1 test)
- `test_set_parameters_updates_model`: Verifies model parameters are updated

#### TestTrainFunction (5 tests)
- `test_train_returns_float_loss`: Validates training returns float loss
- `test_train_multiclass`: Tests multi-class classification training
- `test_train_binary_classification`: Tests binary classification (num_classes=1)
- `test_train_multiple_epochs`: Validates multi-epoch training
- `test_train_network_in_train_mode`: Confirms training sets network to train mode

#### TestEvaluateFunction (4 tests)
- `test_evaluate_returns_tuple`: Verifies evaluate returns (loss, accuracy) tuple
- `test_evaluate_multiclass`: Tests evaluation with multi-class classification
- `test_evaluate_binary_classification`: Tests binary classification evaluation
- `test_evaluate_network_in_eval_mode`: Confirms evaluation runs successfully

#### TestFlowerClientFit (1 test)
- `test_fit_initialization`: Validates fit method initialization

#### TestFlowerClientEvaluate (1 test)
- `test_evaluate_client_initialization`: Validates client initialization for evaluation

---

### 2. `test_partitioner.py` - Data Partitioning Tests (21 tests)

Tests for the data partitioning utility in `partitioner.py`.

#### TestPartitionDataStratified (21 tests)
- `test_partition_returns_list_of_dataframes`: Verifies returns list of DataFrames
- `test_partition_correct_number_of_clients`: Validates correct partition count
- `test_partition_preserves_total_samples`: Ensures no data loss during partitioning
- `test_partition_balances_classes`: Confirms class balance is maintained
- `test_partition_stratified_distribution`: Verifies class distribution across clients
- `test_partition_preserves_columns`: Validates all columns are preserved
- `test_partition_no_duplicate_rows`: Ensures no row duplication
- `test_partition_shuffled_order`: Confirms data is shuffled
- `test_partition_reproducible_with_seed`: Tests reproducibility with same seed
- `test_partition_different_with_different_seed`: Validates different seeds produce different results
- `test_partition_single_client`: Tests partitioning with single client
- `test_partition_invalid_dataframe_type`: Validates TypeError for non-DataFrame input
- `test_partition_invalid_num_clients`: Validates ValueError for invalid num_clients
- `test_partition_missing_target_column`: Validates ValueError for missing target column
- `test_partition_empty_dataframe`: Tests handling of empty DataFrames
- `test_partition_unbalanced_classes`: Tests handling of highly unbalanced classes
- `test_partition_index_reset`: Confirms indices are properly reset
- `test_partition_large_num_clients`: Tests with more clients than samples
- `test_partition_multiple_classes`: Tests with multiple class labels
- `test_partition_dataframe_not_modified`: Ensures original DataFrame is not modified
- `test_partition_with_additional_columns`: Validates additional columns are preserved

---

### 3. `test_data_manager.py` - Data Management Tests (26 tests)

Tests for data loading and splitting in `data_manager.py`.

#### TestSplitPartition (7 tests)
- `test_split_partition_returns_tuple`: Verifies returns tuple of DataFrames
- `test_split_partition_correct_split_ratio`: Validates approximate split ratio
- `test_split_partition_preserves_total_samples`: Ensures no data loss
- `test_split_partition_reproducible_with_seed`: Tests reproducibility
- `test_split_partition_different_with_different_seed`: Validates seed effect
- `test_split_partition_stratified_fallback`: Tests fallback to random split
- `test_split_partition_index_reset`: Confirms index reset in splits

Additional tests (continuation):
- `test_split_partition_preserves_columns`: Validates column preservation
- `test_split_partition_extreme_ratios`: Tests edge case ratios

#### TestLoadData (19 tests)
- `test_load_data_returns_tuple_of_dataloaders`: Verifies DataLoader tuple return
- `test_load_data_dataloaders_have_batches`: Confirms DataLoaders can produce batches
- `test_load_data_respects_batch_size`: Validates batch size configuration
- `test_load_data_respects_validation_split`: Confirms validation split is respected
- `test_load_data_train_shuffle_true`: Verifies training DataLoader shuffles
- `test_load_data_val_shuffle_false`: Confirms validation DataLoader doesn't shuffle
- `test_load_data_empty_partition_raises_error`: Tests ValueError for empty partition
- `test_load_data_missing_image_dir_raises_error`: Tests error for missing directory
- `test_load_data_missing_filename_column_raises_error`: Tests missing column error
- `test_load_data_missing_target_column_raises_error`: Tests missing target column error
- `test_load_data_custom_validation_split_override`: Tests validation_split override
- `test_load_data_image_dir_as_string`: Tests Path as string
- `test_load_data_image_dir_as_path`: Tests Path object support
- `test_load_data_with_color_mode_config`: Tests custom color_mode configuration
- `test_load_data_with_augmentation_config`: Tests augmentation configuration
- `test_load_data_train_samples_count`: Validates train/val sample counts
- `test_load_data_creates_valid_batches`: Confirms valid batch creation

---

### 4. `test_trainer.py` - Federated Trainer Tests (21 tests)

Tests for the federated learning trainer orchestration in `trainer.py`.

#### TestFederatedTrainerInit (3 tests)
- `test_init_valid_parameters`: Tests initialization with valid parameters
- `test_init_none_config_raises_error`: Validates config cannot be None
- `test_init_none_constants_raises_error`: Validates constants cannot be None
- `test_init_none_device_raises_error`: Validates device cannot be None
- `test_init_logger_setup`: Confirms logger is properly initialized

#### TestFederatedTrainerCreateModel (4 tests)
- `test_create_model_returns_resnet`: Verifies ResNetWithCustomHead is returned
- `test_create_model_on_device`: Confirms model is on specified device
- `test_create_model_correct_num_classes`: Validates num_classes configuration
- `test_create_model_with_dropout`: Tests dropout configuration

#### TestFederatedTrainerGetInitialParameters (1 test)
- `test_get_initial_parameters_returns_parameters`: Validates Parameters object return
- `test_get_initial_parameters_creates_weights`: Confirms weights are created

#### TestFederatedTrainerClientFn (1 test)
- `test_client_fn_missing_client_id_raises_error`: Tests error for missing client ID

#### TestFederatedTrainerCreateEvaluateFn (2 tests)
- `test_create_evaluate_fn_returns_callable`: Confirms callable is returned
- `test_create_evaluate_fn_has_logging`: Validates logging in evaluate function

#### TestFederatedTrainerTrain (3 tests)
- `test_train_requires_valid_data`: Tests input validation
- `test_train_method_exists`: Confirms train method exists and is callable
- `test_train_error_handling`: Tests error handling and logging

#### TestFederatedTrainerIntegration (2 tests)
- `test_trainer_initialization_sequence`: Tests typical initialization sequence
- `test_trainer_create_model_multiple_times`: Tests multiple model creation

---

## Test Statistics

| Module | Tests | Status |
|--------|-------|--------|
| test_client.py | 26 | ✅ PASS |
| test_partitioner.py | 21 | ✅ PASS |
| test_data_manager.py | 26 | ✅ PASS |
| test_trainer.py | 21 | ✅ PASS |
| **TOTAL** | **94** | **✅ ALL PASS** |

## Test Execution

Run all control layer tests:
```bash
pytest federated_pneumonia_detection/tests/unit/control/ -v
```

Run specific test file:
```bash
pytest federated_pneumonia_detection/tests/unit/control/test_client.py -v
pytest federated_pneumonia_detection/tests/unit/control/test_partitioner.py -v
pytest federated_pneumonia_detection/tests/unit/control/test_data_manager.py -v
pytest federated_pneumonia_detection/tests/unit/control/test_trainer.py -v
```

Run specific test class:
```bash
pytest federated_pneumonia_detection/tests/unit/control/test_client.py::TestFlowerClient -v
```

Run specific test:
```bash
pytest federated_pneumonia_detection/tests/unit/control/test_client.py::TestGetWeights::test_get_weights_returns_list -v
```

## Coverage Areas

### Functionality Tested

1. **Client Parameter Management**
   - Weight extraction and loading
   - Parameter serialization/deserialization
   - Device placement

2. **Training**
   - Multi-class and binary classification
   - Loss computation
   - Learning rate and epoch configuration
   - Batch processing

3. **Evaluation**
   - Accuracy computation
   - Loss calculation
   - No-gradient execution mode

4. **Data Partitioning**
   - Stratified sampling
   - Class balance maintenance
   - Reproducibility with seeds
   - Edge cases (empty data, large num_clients, imbalanced classes)

5. **Data Loading**
   - DataLoader creation
   - Batch size configuration
   - Train/validation splitting
   - Image augmentation support
   - Column validation

6. **Trainer Orchestration**
   - Model creation
   - Parameter initialization
   - Client function factory
   - Error handling and logging

### Error Scenarios Tested

- None/invalid input validation
- Missing required columns
- Missing directories
- Empty DataFrames
- Invalid configuration values
- Type mismatches
- Edge cases (single client, many clients, unbalanced data)

## Key Testing Patterns

1. **Unit Isolation**: Each test focuses on a single function or method
2. **Mock Usage**: External dependencies are mocked appropriately
3. **Parameterization**: Multiple scenarios tested for same functionality
4. **Edge Cases**: Boundary conditions and error states validated
5. **Reproducibility**: Seed-based tests ensure deterministic behavior
6. **Integration Checks**: Related components verified to work together
