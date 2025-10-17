# ClientDataManager - Quick Reference

## TL;DR

Convert DataFrame partitions into DataLoaders for federated clients.

```python
from federated_pneumonia_detection.src.control.federated_learning.data_manager import ClientDataManager
from pathlib import Path

# Setup (once per client)
data_manager = ClientDataManager(
    image_dir=Path('./images'),
    constants=constants,
    config=config
)

# Create DataLoaders
train_loader, val_loader = data_manager.create_dataloaders_for_partition(
    partition_df=client_partition  # DataFrame with 'filename' and 'Target'
)

# Use in training
for images, labels in train_loader:
    # Train step
    pass
```

## File Location

```
C:\Users\User\Projects\FYP2\
  federated_pneumonia_detection/src/control/federated_learning/
    data_manager.py
```

## Class: ClientDataManager

### Constructor

```python
ClientDataManager(
    image_dir: Union[str, Path],
    constants: SystemConstants,
    config: ExperimentConfig,
    logger: Optional[logging.Logger] = None
)
```

| Parameter | Type | Purpose |
|-----------|------|---------|
| `image_dir` | Path | Directory with images |
| `constants` | SystemConstants | IMG_SIZE, column names |
| `config` | ExperimentConfig | batch_size, validation_split, augmentation |
| `logger` | Logger | Optional logging |

**Raises:** `ValueError` if image_dir invalid

### Main Method

```python
create_dataloaders_for_partition(
    partition_df: pd.DataFrame,
    validation_split: Optional[float] = None
) -> Tuple[DataLoader, DataLoader]
```

**Parameters:**
- `partition_df`: DataFrame with columns 'filename' and 'Target'
- `validation_split`: Override config.validation_split (optional)

**Returns:** (train_loader, val_loader)

**Raises:**
- `ValueError`: Empty partition or missing columns
- `RuntimeError`: Dataset creation failed

## Partition DataFrame Format

```python
partition_df = pd.DataFrame({
    'filename': ['img_001.png', 'img_002.png', 'img_003.png'],
    'Target': [0, 1, 0]  # 0=Normal, 1=Pneumonia
})
```

**Requirements:**
- Column names match `constants.FILENAME_COLUMN` ('filename')
- Column names match `constants.TARGET_COLUMN` ('Target')
- At least 2 rows recommended (stratified split needs it)
- All image files must exist in `image_dir`

## Common Configurations

### Minimal Setup

```python
constants = SystemConstants()
config = ExperimentConfig()

manager = ClientDataManager(
    image_dir=Path('./images'),
    constants=constants,
    config=config
)
```

### Training Setup (with Augmentation)

```python
config = ExperimentConfig(
    batch_size=128,
    validation_split=0.2,
    augmentation_strength=1.0,  # Increase for more augmentation
    pin_memory=True,  # GPU transfer (not Windows Flower)
)

manager = ClientDataManager(
    image_dir=Path('./images'),
    constants=constants,
    config=config
)
```

### Custom Image Size

```python
constants = SystemConstants(IMG_SIZE=(256, 256))
config = ExperimentConfig(batch_size=64)

manager = ClientDataManager(
    image_dir=Path('./images'),
    constants=constants,
    config=config
)
```

### X-ray Preprocessing

```python
config = ExperimentConfig(
    use_custom_preprocessing=True,
    contrast_stretch=True,
    adaptive_histogram=False,
    edge_enhancement=False,
)

manager = ClientDataManager(
    image_dir=Path('./images'),
    constants=constants,
    config=config
)
```

## Example: Federated Client

```python
from federated_pneumonia_detection.src.control.federated_learning.data_manager import ClientDataManager
from pathlib import Path

class FederatedClient:
    def __init__(self, client_id, partition_df, config, constants):
        # Create manager ONCE
        self.manager = ClientDataManager(
            image_dir=Path(f'./federated_data/{client_id}'),
            constants=constants,
            config=config
        )

        # Create DataLoaders ONCE
        self.train_loader, self.val_loader = \
            self.manager.create_dataloaders_for_partition(partition_df)

    def train_epoch(self, model):
        for images, labels in self.train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def evaluate(self, model):
        for images, labels in self.val_loader:
            outputs = model(images)
            metrics.update(outputs, labels)
```

## Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Image directory not found` | Path doesn't exist | Check `image_dir` path |
| `ValueError: partition_df cannot be empty` | Empty DataFrame | Ensure partition has rows |
| `ValueError: Missing required columns` | Wrong column names | Use 'filename' and 'Target' |
| `RuntimeError: Dataset creation failed` | Bad images or transforms | Check image files exist |
| `Stratification failed, falling back` | <2 unique labels | Normal - uses random split |

## Data Flow

```
Partition DataFrame (100 rows)
    ↓
_split_partition()
    ├─ Stratified split (80 train, 20 val)
    ├─ Fallback to random if needed
    └─ Reset indices
    ↓
build_training_transforms() + build_validation_transforms()
    ├─ Resize, augmentation, normalization
    └─ Same for both, but val has less augmentation
    ↓
CustomImageDataset (train: 80 samples, val: 20 samples)
    └─ Load images, apply transforms
    ↓
DataLoader (train: 1 batch of 80, val: 1 batch of 20)
    └─ batch_size=128, shuffle=True/False
    ↓
Return (train_loader, val_loader)
```

## Performance Tips

1. **Create manager ONCE per client** - avoid recreating TransformBuilder
2. **Set `validate_images=False`** in config - skip image validation for speed
3. **Use `num_workers=0`** - required for Windows/Flower (already set)
4. **Use `pin_memory=False`** on Windows Flower - memory issues with True
5. **Cache partition DataFrame** - don't recreate each round

## Integration Points

### With Flower Client
```python
# In FlowerClient.__init__
self.train_loader, self.val_loader = \
    ClientDataManager(...).create_dataloaders_for_partition(partition)
```

### With Data Distributor
```python
# After partitioning
for client_id, partition in partitions.items():
    manager = ClientDataManager(...)
    train_loader, val_loader = manager.create_dataloaders_for_partition(partition)
```

### With Config System
```python
# All settings come from ExperimentConfig
config = ExperimentConfig(
    batch_size=...,           # Used by DataManager
    validation_split=...,     # Used by DataManager
    augmentation_strength=... # Used by DataManager
)
```

## Design Principles

| Principle | Implementation |
|-----------|-----------------|
| SRP | One job: create DataLoaders |
| OCP | Extensible via config injection |
| LSP | Works with any PyTorch Dataset |
| ISP | Focused interface |
| DIP | Depends on abstractions (config, constants) |

## Key Features

- Stratified train/val split (with fallback)
- Automatic transform building
- Windows/Flower compatibility
- Comprehensive error handling
- Configurable augmentation strength
- X-ray preprocessing support
- Memory-efficient caching

## What It Does NOT Do

- Create data partitions (use DataDistributor)
- Handle model training (use FlowerClient)
- Download/prepare images (do this before)
- Aggregate weights (use Flower server)
- Track experiments (use ExperimentTracker)

## Dependencies

| Dependency | Role |
|-----------|------|
| pandas | DataFrame manipulation |
| sklearn | Stratified splitting |
| PyTorch | DataLoader |
| TransformBuilder | Image augmentation |
| CustomImageDataset | Image loading |

---

**See Also:**
- Full docs: `CLIENT_DATA_MANAGER_ARCHITECTURE.md`
- Examples: `CLIENT_DATA_MANAGER_EXAMPLE.py`
- Integration: `CLIENT_DATA_MANAGER_INTEGRATION.md`
