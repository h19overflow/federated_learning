# Phase 1 Quick Start Guide

**Get up and running with the federated pneumonia detection system in 5 minutes**

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Basic understanding of machine learning workflows

## Installation

```bash
# Navigate to project directory
cd federated_pneumonia_detection

# Install dependencies
uv pip install -r requirements.txt
# OR
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Data Pipeline

Create a simple script to test the data pipeline:

```python
# quick_test.py
import logging
from src.entities import SystemConstants, ExperimentConfig
from src.utils import load_and_split_data, get_image_directory_path

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Create configuration
constants = SystemConstants.create_custom(
    sample_fraction=0.1,  # Use 10% of data for quick testing
    batch_size=32,
    seed=42
)

config = ExperimentConfig(
    sample_fraction=0.1,
    validation_split=0.2,
    batch_size=32
)

print("Configuration created successfully!")
print(f"Image size: {constants.IMG_SIZE}")
print(f"Batch size: {config.batch_size}")

# Test data loading (will fail if no data files exist, but shows the API)
try:
    train_df, val_df = load_and_split_data(constants, config)
    print(f"Data loaded: {len(train_df)} train, {len(val_df)} validation samples")
except FileNotFoundError as e:
    print(f"Data files not found (expected): {e}")
    print("✅ Pipeline structure is working - you just need actual data files")
```

### 2. Configuration from YAML

```python
# config_test.py
from src.utils import ConfigLoader

# Load from YAML configuration
config_loader = ConfigLoader()

try:
    constants = config_loader.create_system_constants()
    config = config_loader.create_experiment_config()

    print("✅ Configuration loaded from YAML")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")
    print(f"Model: ResNet50 with dropout {config.dropout_rate}")

except FileNotFoundError:
    print("⚠️ No config/default_config.yaml found")
    print("Using default configuration instead")
```

### 3. Complete Pipeline Test

```python
# full_pipeline_test.py
from src.entities import SystemConstants, ExperimentConfig
from src.control import XRayDataModule, LitResNet
from src.utils import load_and_split_data, get_image_directory_path
import pandas as pd

# Create minimal test configuration
constants = SystemConstants.create_custom(
    sample_fraction=0.05,  # Very small for quick testing
    batch_size=16
)

config = ExperimentConfig(
    sample_fraction=0.05,
    batch_size=16,
    dropout_rate=0.3,
    use_custom_preprocessing=True
)

print("=== Testing Complete Pipeline ===")

# Test 1: Model Creation
print("\n1. Testing model creation...")
model = LitResNet(constants=constants, config=config)
model_info = model.get_model_summary()
print(f"✅ Model created: {model_info['total_parameters']} parameters")

# Test 2: Data Module (without actual data)
print("\n2. Testing data module creation...")
# Create dummy DataFrames for testing
dummy_train = pd.DataFrame({
    'patientId': ['001', '002', '003'],
    'Target': [0, 1, 0],
    'filename': ['001.png', '002.png', '003.png']
})
dummy_val = pd.DataFrame({
    'patientId': ['004'],
    'Target': [1],
    'filename': ['004.png']
})

image_dir = get_image_directory_path(constants)
data_module = XRayDataModule(
    train_df=dummy_train,
    val_df=dummy_val,
    constants=constants,
    config=config,
    image_dir=image_dir,
    validate_images_on_init=False  # Skip validation for dummy data
)

stats = data_module.get_data_statistics()
print(f"✅ Data module created: {stats['train_samples']} train samples")

print("\n🎉 All components working correctly!")
print("Ready for actual data and training!")
```

## Data Setup

To work with real data, organize your files like this:

```
your_data_directory/
├── Images/
│   └── Images/
│       ├── patient_001.png
│       ├── patient_002.png
│       └── ...
└── Train_metadata.csv
```

**Train_metadata.csv format:**
```csv
patientId,Target,age,gender
patient_001,0,45,M
patient_002,1,52,F
patient_003,0,38,F
```

Update your configuration:
```python
constants = SystemConstants.create_custom(
    base_path="/path/to/your/data",
    metadata_filename="Train_metadata.csv"
)
```

## Running the Complete Example

```bash
# Run the complete pipeline example
python examples/complete_data_pipeline_example.py
```

This will show you the entire data flow from CSV to ready-to-train datasets.

## Common Configuration Options

### Quick Experimentation
```python
config = ExperimentConfig(
    sample_fraction=0.1,      # Use 10% of data
    batch_size=32,            # Smaller batches
    epochs=5,                 # Quick training
    learning_rate=0.01        # Higher LR for faster convergence
)
```

### Production Setup
```python
config = ExperimentConfig(
    sample_fraction=1.0,      # Use all data
    batch_size=128,           # Larger batches
    epochs=50,                # Full training
    learning_rate=0.001,      # Standard LR
    early_stopping_patience=10
)
```

### Fine-tuning Setup
```python
config = ExperimentConfig(
    freeze_backbone=False,           # Unfreeze backbone
    fine_tune_layers_count=-5,       # Unfreeze last 5 layers
    learning_rate=0.0001,            # Lower LR for fine-tuning
    use_custom_preprocessing=True    # X-ray specific preprocessing
)
```

## Environment Variables

Override any configuration with environment variables:

```bash
# Override learning rate and batch size
export FPD_LEARNING_RATE=0.01
export FPD_BATCH_SIZE=64
export FPD_USE_CUSTOM_PREPROCESSING=true

python your_script.py
```

## Next Steps

1. **Add Your Data**: Organize your X-ray images and metadata CSV
2. **Customize Configuration**: Modify `config/default_config.yaml`
3. **Test Training**: Use PyTorch Lightning Trainer with your LitResNet model
4. **Monitor Results**: Check metrics and model performance

## Troubleshooting

### Common Issues

**"Configuration file not found"**
```python
# Create default config or use manual configuration
constants = SystemConstants()  # Uses defaults
```

**"Metadata file not found"**
```python
# Check your data path
constants = SystemConstants.create_custom(
    base_path="/correct/path/to/your/data"
)
```

**"No valid images found"**
- Check image directory structure: `Images/Images/*.png`
- Verify image file names match metadata `patientId` column
- Check file permissions and formats

**"CUDA out of memory"**
```python
# Reduce batch size
config = ExperimentConfig(batch_size=16)
# Or use CPU
config = ExperimentConfig(device='cpu')
```

### Getting Help

1. **Check Logs**: All components have detailed logging
2. **Validate Configuration**: Use the validation methods
3. **Test Components**: Each component can be tested independently
4. **Check Examples**: See `examples/` directory for working code

## Example Output

When everything is working correctly, you should see:

```
INFO - Configuration loaded from config/default_config.yaml
INFO - Loaded metadata: 1000 samples from data/Train_metadata.csv
INFO - Stratified sampling: 100 samples
INFO - Created train/val split - Train: 80, Val: 20
INFO - DataModule created
INFO - Train dataset created: 75 valid samples
INFO - Val dataset created: 18 valid samples
INFO - LitResNet initialized with 23,518,273 parameters
✅ Pipeline ready for training!
```

You're now ready to move to Phase 2 or start training with your data!
