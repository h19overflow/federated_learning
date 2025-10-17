# ClientDataManager Integration Guide

## System Context

ClientDataManager is a utility component that bridges data partitions and model training in the federated learning pipeline.

```
Data Distribution
      ↓
Client Partition (DataFrame)
      ↓
ClientDataManager ← (uses) TransformBuilder, CustomImageDataset
      ↓
DataLoaders (train + validation)
      ↓
Flower Client (federated_trainer.py)
      ↓
Model Training
```

## Integration Points

### 1. With Data Distribution Pipeline

**Scenario:** Federated Data Distributor creates per-client partitions

```python
from federated_pneumonia_detection.src.control.federated_learning.data_manager import ClientDataManager

# Data distributor creates partition for each client
client_partitions = distribute_data_to_clients(full_dataset, num_clients=5)

# Client receives its partition
client_partition = client_partitions['client_001']  # DataFrame

# ClientDataManager converts partition to DataLoaders
data_manager = ClientDataManager(
    image_dir=Path('./images'),
    constants=constants,
    config=config
)

train_loader, val_loader = data_manager.create_dataloaders_for_partition(client_partition)
```

### 2. With Flower Client Implementation

**Where:** In `ClientApp` or `FlowerClient` class

```python
# flower_client.py
from federated_pneumonia_detection.src.control.federated_learning.data_manager import ClientDataManager

class PneumoniaFlowerClient:
    def __init__(self, client_id, data_partition, config):
        self.client_id = client_id
        self.data_partition = data_partition
        self.config = config

        # Initialize data manager ONCE
        self.data_manager = ClientDataManager(
            image_dir=Path(f'./federated_data/{client_id}/images'),
            constants=SystemConstants(),
            config=config,
            logger=get_logger(__name__)
        )

        # Create DataLoaders (can be called multiple times per round)
        self.train_loader, self.val_loader = \
            self.data_manager.create_dataloaders_for_partition(data_partition)

    def fit(self, model, epochs):
        """Local training round using DataLoaders."""
        for epoch in range(epochs):
            for images, labels in self.train_loader:
                # Train step
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate(self, model):
        """Validation using DataLoader."""
        model.eval()
        for images, labels in self.val_loader:
            outputs = model(images)
            # Compute metrics
```

### 3. With Config System

**Configuration propagates through DataManager:**

```python
# experiment_config.py
config = ExperimentConfig(
    batch_size=128,                    # Used by DataManager → DataLoader
    validation_split=0.2,              # Used by DataManager → _split_partition
    augmentation_strength=1.0,         # Used by DataManager → TransformBuilder
    use_custom_preprocessing=False,    # Used by DataManager → TransformBuilder
    color_mode='RGB',                  # Used by DataManager → CustomImageDataset
    pin_memory=True,                   # Used by DataManager → DataLoader
)

# DataManager receives config and applies all settings
data_manager = ClientDataManager(..., config=config, ...)
```

### 4. With Transform System

**DataManager orchestrates transform creation:**

```
TransformBuilder (receives constants + config)
    ├─ build_training_transforms()
    │  └─ With augmentation (random crop, rotation, color jitter)
    │  └─ With optional custom preprocessing
    │  └─ With normalization
    │
    └─ build_validation_transforms()
       └─ Without augmentation (simple resize + crop)
       └─ With optional custom preprocessing
       └─ With same normalization
```

DataManager calls both and applies to appropriate datasets:

```python
train_transform = self.transform_builder.build_training_transforms(...)
val_transform = self.transform_builder.build_validation_transforms(...)

train_dataset = CustomImageDataset(..., transform=train_transform)
val_dataset = CustomImageDataset(..., transform=val_transform)
```

## Usage in Federated Training

### 1. Client Initialization

```python
def create_client(client_id, client_partition, config):
    """Factory function for creating a federated client."""

    data_manager = ClientDataManager(
        image_dir=Path(f'./federated_data/{client_id}/images'),
        constants=SystemConstants(),
        config=config,
        logger=get_logger(f'client_{client_id}')
    )

    train_loader, val_loader = data_manager.create_dataloaders_for_partition(
        partition_df=client_partition
    )

    return FlowerClient(
        client_id=client_id,
        train_loader=train_loader,
        val_loader=val_loader,
        model=load_pretrained_model(),
        config=config
    )
```

### 2. Per-Round Training

```python
def fit_round(client, num_epochs):
    """Perform one federated training round."""

    # DataLoaders already created (via ClientDataManager)
    train_loader = client.train_loader

    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = client.model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### 3. Per-Round Evaluation

```python
def evaluate_round(client):
    """Evaluate client model on validation set."""

    # DataLoaders already created (via ClientDataManager)
    val_loader = client.val_loader

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = client.model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)

    return {'accuracy': accuracy, 'loss': avg_loss}
```

## Data Flow Example

### Round 1: Client 1 Training

```
Input: Client 1 receives 500 samples with labels

ClientDataManager._split_partition(partition_df, validation_split=0.2)
    ├─ Stratified split: 400 train, 100 validation
    ├─ Reset indices
    └─ Returns (train_df, val_df)

create_dataloaders_for_partition(partition_df)
    ├─ Split partition
    ├─ Build training transforms (with augmentation)
    ├─ Build validation transforms (without augmentation)
    ├─ Create CustomImageDataset for train (400 samples)
    │  └─ With training transforms
    │  └─ Validate images = False (for speed)
    ├─ Create CustomImageDataset for validation (100 samples)
    │  └─ With validation transforms
    │  └─ Validate images = False
    ├─ Create DataLoader(train_dataset, batch_size=128, shuffle=True)
    │  └─ num_batches = ceil(400/128) = 4 batches
    ├─ Create DataLoader(val_dataset, batch_size=128, shuffle=False)
    │  └─ num_batches = ceil(100/128) = 1 batch
    └─ Returns (train_loader, val_loader)

Flower Client Training Loop:
    for epoch in range(1):
        for batch in train_loader:  # 4 batches
            # Train step

    for batch in val_loader:  # 1 batch
        # Validation step

Output: Trained model weights for aggregation
```

## Key Design Decisions for Integration

### 1. Single Manager Per Client

Create ONE ClientDataManager per client, reuse for all rounds:

```python
# GOOD: Create once in __init__
self.data_manager = ClientDataManager(...)
self.train_loader, self.val_loader = \
    self.data_manager.create_dataloaders_for_partition(partition)

# BAD: Don't create new manager each round
for round in rounds:
    manager = ClientDataManager(...)  # Recreates TransformBuilder!
    train_loader, val_loader = manager.create_dataloaders_for_partition(partition)
```

### 2. Partition Stays Static Per Client

Each client's partition is fixed for the entire federated learning process:

```python
# Distribute once at setup
client_partitions = federated_data_distributor.distribute(...)

# Each client keeps same partition across all rounds
for round in num_rounds:
    # Client 1 always uses client_partitions['client_1']
    train_loader, val_loader = manager.create_dataloaders_for_partition(
        client_partitions['client_1']
    )
```

### 3. No Cross-Client Data Sharing

Each client ONLY sees its partition:

```python
# Client 1
manager_1 = ClientDataManager('client_1/images', ...)
train_loader_1, val_loader_1 = manager_1.create_dataloaders_for_partition(
    partition_1
)

# Client 2 (different images, different labels)
manager_2 = ClientDataManager('client_2/images', ...)
train_loader_2, val_loader_2 = manager_2.create_dataloaders_for_partition(
    partition_2
)

# No interaction between clients
```

## Testing Integration

### Unit Test: ClientDataManager in Isolation

```python
def test_data_manager_creates_correct_loaders():
    manager = ClientDataManager(...)
    train_loader, val_loader = manager.create_dataloaders_for_partition(partition_df)

    assert len(train_loader) == 4
    assert len(val_loader) == 1
```

### Integration Test: ClientDataManager with FlowerClient

```python
def test_flower_client_with_data_manager():
    partition = create_test_partition(100)
    client = PneumoniaFlowerClient('client_1', partition, config)

    # Should be able to train
    model = create_test_model()
    client.fit(model, epochs=1)

    # Should be able to evaluate
    metrics = client.evaluate(model)
    assert 'loss' in metrics
    assert 'accuracy' in metrics
```

### End-to-End Test: Full Federated Round

```python
def test_federated_round():
    # Setup
    config = ExperimentConfig()
    constants = SystemConstants()

    # Distribute data
    partitions = distribute_data(dataset, num_clients=3)

    # Create clients with DataManagers
    clients = [
        PneumoniaFlowerClient(f'client_{i}', partitions[i], config)
        for i in range(3)
    ]

    # Perform federated round
    model = create_model()
    for client in clients:
        client.fit(model, epochs=1)
        metrics = client.evaluate(model)

    # Aggregate (in real setup)
    # updated_model = federated_averaging(clients)
```

## Performance Considerations

### Memory Usage

- **One DataLoader:** ~500MB for 5000 images (224x224 RGB)
- **Two DataManagers:** ~1GB (train + val loaders)
- **Per Client:** Minimal overhead

### Computation

- **TransformBuilder creation:** One-time cost (~10ms)
- **Per batch loading:** ~50-100ms (depends on num_workers setting)
- **Dataset creation:** O(n) where n = partition size

### Optimization Tips

1. **Use `pin_memory=True`** for GPU training (if not on Windows)
2. **Use `num_workers=0`** on Windows (Flower limitation)
3. **Set `validate_images=False`** for faster initialization
4. **Cache transforms** (already done in `__init__`)

## Troubleshooting

### Issue: "Image directory not found"

```python
# Check path exists
image_dir = Path('./federated_data/client_1/images')
assert image_dir.exists(), f"Missing: {image_dir}"

# Create if needed
image_dir.mkdir(parents=True, exist_ok=True)
```

### Issue: "Missing required columns"

```python
# Ensure DataFrame has correct column names
partition_df = pd.DataFrame({
    'filename': [f'img_{i}.png' for i in range(100)],  # Constants.FILENAME_COLUMN
    'Target': [i % 2 for i in range(100)]              # Constants.TARGET_COLUMN
})
```

### Issue: "Dataset creation failed"

```python
# Check images exist and are valid
for filename in partition_df['filename'].head(10):
    img_path = image_dir / filename
    assert img_path.exists(), f"Missing: {img_path}"

    # Try opening
    from PIL import Image
    img = Image.open(img_path)
    img.verify()
```

### Issue: "Stratification failed"

This is normal and automatically handled. DataManager falls back to random split:

```python
# Log shows:
# "Stratification failed, falling back to random split"

# This happens when:
# - Partition has <2 samples per class
# - Only 1 class in partition
# - Class distribution too imbalanced
```

---

**For more details, see:**
- `CLIENT_DATA_MANAGER_ARCHITECTURE.md` - Design patterns and SOLID principles
- `CLIENT_DATA_MANAGER_EXAMPLE.py` - Usage examples
- Source: `federated_pneumonia_detection/src/control/federated_learning/data_manager.py`
