# ClientDataManager Architecture

## Overview

`ClientDataManager` is a focused utility component for creating PyTorch DataLoaders from individual client data partitions in federated learning scenarios. It encapsulates data loading logic, stratified splitting, and transform management into a single, reusable interface.

**File Location:** `federated_pneumonia_detection/src/control/federated_learning/data_manager.py`

## Component Responsibility

Single Responsibility: Create train/validation DataLoaders from a client's data partition.

```
Input: DataFrame partition + config → Output: (train_loader, val_loader)
```

## Architecture Decisions

### 1. Constructor Pattern: Build Expensive Objects Once

**Why:** TransformBuilder involves model loading and augmentation pipeline creation. These are expensive operations.

```python
def __init__(self, image_dir, constants, config, logger):
    self.transform_builder = TransformBuilder(constants, config)  # Once!
    # ...
```

**Impact:** Reusing the same TransformBuilder across multiple `create_dataloaders_for_partition` calls avoids redundant computation.

### 2. Dependency Injection via Constructor

All dependencies are injected, not created internally:
- `SystemConstants`: Image size, batch size, column names
- `ExperimentConfig`: Augmentation settings, validation split, pin_memory
- `logger`: Optional logging instance

**Benefit:** Components are testable and loosely coupled. Easy to mock for testing.

### 3. Composition Over Inheritance

ClientDataManager composes:
- `TransformBuilder` (for creating augmentation pipelines)
- `CustomImageDataset` (for loading images)
- `DataLoader` (PyTorch native)

**Why:** No inheritance needed. Clear, focused interfaces. Each component does one thing.

### 4. Fail-Fast Validation

Input validation happens immediately at entry points:

```python
if partition_df.empty:
    raise ValueError("partition_df cannot be empty")

required_cols = [self.constants.FILENAME_COLUMN, self.constants.TARGET_COLUMN]
missing_cols = [col for col in required_cols if col not in partition_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
```

**Benefit:** Errors are caught early with clear context. No silent failures.

### 5. Stratified Split with Fallback

Attempts stratified split (preserves class distribution), falls back to random:

```python
try:
    # Stratified by Target column
    train_df, val_df = train_test_split(..., stratify=partition_df[TARGET])
except (ValueError, TypeError):
    # Fallback: insufficient unique values or other issues
    train_df, val_df = train_test_split(...)
```

**Why:** Stratification ensures balanced validation sets when possible. Fallback handles edge cases (single-class partitions, very small datasets).

### 6. Configuration Over Hardcoding

All tunable values come from injected config:

```python
batch_size = self.config.batch_size
augmentation_strength = self.config.augmentation_strength
use_custom_preprocessing = self.config.use_custom_preprocessing
pin_memory = self.config.pin_memory
color_mode = self.config.color_mode
```

**Benefit:** Same component works across different experiments without modification.

### 7. Windows/Flower Limitations

```python
num_workers=0,  # Windows/Flower limitation
drop_last=True  # Train only (drop incomplete batches)
drop_last=False # Validation (use all samples)
persistent_workers=False  # Windows compatibility
```

**Why:** Flower framework on Windows doesn't support multiprocessing for DataLoaders. `persistent_workers` causes issues on Windows.

## Class Interface

### `__init__(image_dir, constants, config, logger=None)`

Initializes the manager. Validates image directory and creates TransformBuilder once.

**Parameters:**
- `image_dir` (str | Path): Directory containing client's images
- `constants` (SystemConstants): System configuration (IMG_SIZE, column names, etc.)
- `config` (ExperimentConfig): Experiment parameters (batch_size, augmentation, etc.)
- `logger` (Optional[logging.Logger]): Logger instance

**Raises:**
- `ValueError`: If image_dir doesn't exist or isn't a directory

### `create_dataloaders_for_partition(partition_df, validation_split=None)`

Main public method. Creates train and validation DataLoaders.

**Parameters:**
- `partition_df` (pd.DataFrame): DataFrame with 'filename' and 'Target' columns
- `validation_split` (Optional[float]): Override config.validation_split (default: None)

**Returns:**
- `Tuple[DataLoader, DataLoader]`: (train_loader, val_loader)

**Raises:**
- `ValueError`: If partition_df is empty or missing required columns
- `RuntimeError`: If dataset creation fails

**Flow:**
1. Validate partition DataFrame
2. Split into train/validation using stratified split with fallback
3. Build training and validation transforms
4. Create CustomImageDataset instances
5. Create DataLoaders with proper settings
6. Return tuple of loaders

### `_split_partition(partition_df, validation_split)` (Private)

Helper method for stratified splitting.

**Why Private:** Internal implementation detail. Users call `create_dataloaders_for_partition`, not this.

**Behavior:**
1. Attempts stratified split by Target column
2. Falls back to random split if stratification fails
3. Resets indices for consistency

## Design Patterns Applied

### 1. Strategy Pattern (Implicit)

TransformBuilder implements different transform strategies:
- Training (with augmentation)
- Validation (without augmentation)

ClientDataManager doesn't need to know *how* transforms are built, just that it gets different pipelines for train vs. validation.

### 2. Dependency Injection

All external dependencies are passed to constructor:
```python
def __init__(self, ..., constants, config, logger):
    self.constants = constants
    self.config = config
    self.logger = logger
```

**Benefit:** Easy to test with mocks, flexible for different configurations.

### 3. Builder Pattern (via TransformBuilder)

TransformBuilder constructs complex transform pipelines. ClientDataManager uses it without knowing implementation details.

## SOLID Principles

### Single Responsibility Principle (SRP)
- **One job:** Create train/val DataLoaders from a partition
- **Not responsible for:** Model training, evaluation, data distribution, experiment tracking

### Open/Closed Principle (OCP)
- **Open to extension:** Override validation_split per call
- **Closed to modification:** No need to change code for different configs
- How: Dependency injection + configuration parameters

### Liskov Substitution Principle (LSP)
- CustomImageDataset can be swapped for another Dataset implementation
- DataLoader interface is standard PyTorch

### Interface Segregation Principle (ISP)
- Public interface is minimal: just `create_dataloaders_for_partition()`
- Config dependency is focused (ExperimentConfig for data-related settings only)

### Dependency Inversion Principle (DIP)
- Depends on abstractions: SystemConstants, ExperimentConfig, TransformBuilder
- Not tied to concrete file systems or specific augmentation implementations

## Testing Strategy

### Unit Test Example

```python
def test_create_dataloaders_for_partition():
    # Mock dependencies
    constants = SystemConstants()
    config = ExperimentConfig(batch_size=32)
    logger = logging.getLogger(__name__)

    # Create manager with real image directory
    manager = ClientDataManager(
        image_dir=Path('./test_images'),
        constants=constants,
        config=config,
        logger=logger
    )

    # Create test DataFrame
    partition_df = pd.DataFrame({
        'filename': ['test1.png', 'test2.png', 'test3.png'],
        'Target': [0, 1, 0]
    })

    # Create DataLoaders
    train_loader, val_loader = manager.create_dataloaders_for_partition(partition_df)

    # Assertions
    assert len(train_loader) > 0
    assert len(val_loader) > 0

    # Check batch structure
    for images, labels in train_loader:
        assert images.shape[0] <= 32  # batch_size
        assert labels.shape[0] == images.shape[0]
        break
```

### Integration Test Example

```python
def test_federated_client_integration():
    # Simulate federated client
    client_data = load_client_partition()

    manager = ClientDataManager(
        image_dir=Path('./federated_data/client_001'),
        constants=constants,
        config=config
    )

    train_loader, val_loader = manager.create_dataloaders_for_partition(client_data)

    # Train one round
    model = load_model()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
```

## Configuration Parameters Used

From `ExperimentConfig`:
- `batch_size`: DataLoader batch size
- `validation_split`: Fraction of partition for validation
- `augmentation_strength`: Strength of training augmentations
- `use_custom_preprocessing`: Whether to apply X-ray preprocessing
- `color_mode`: 'RGB' or 'L' (grayscale)
- `validate_images_on_init`: Validate images during dataset creation
- `pin_memory`: Whether to pin memory for GPU transfer
- `seed`: Random seed for stratification

From `SystemConstants`:
- `IMG_SIZE`: Target image size (e.g., 224x224)
- `FILENAME_COLUMN`: Column name for filenames
- `TARGET_COLUMN`: Column name for labels

## Error Handling

1. **Invalid image directory:** Raises `ValueError` in `__init__`
2. **Empty partition:** Raises `ValueError` in `create_dataloaders_for_partition`
3. **Missing columns:** Raises `ValueError` in `create_dataloaders_for_partition`
4. **Dataset creation failure:** Catches exceptions, logs, raises `RuntimeError`

All errors include contextual messages for debugging.

## Extension Points (Future Enhancements)

1. **Custom dataset class:** Pass dataset factory as parameter
2. **Custom split strategy:** Support different splitting algorithms
3. **Data caching:** Add optional caching layer for repeated access
4. **Imbalanced sampling:** Add option for weighted/oversampling
5. **Multiple image directories:** Support sharded data across directories

These would be added without modifying existing code (OCP).

## File Size and Complexity

- **Lines of code:** ~225 (including docstrings)
- **Cyclomatic complexity:** Low (straightforward flow)
- **Dependencies:** 5 external + 4 internal
- **Public methods:** 1 (`create_dataloaders_for_partition`)
- **Private methods:** 1 (`_split_partition`)

## Why This Design

1. **Simplicity:** Does one thing well. Easy to understand in 5 minutes.
2. **Reusability:** Works for any client partition without modification.
3. **Testability:** All dependencies injected, no global state.
4. **Maintainability:** Clear responsibilities, explicit error messages, comprehensive docs.
5. **Extensibility:** Design allows for future enhancements without breaking changes.

---

**Created as part of federated pneumonia detection system.**
