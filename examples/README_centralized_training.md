# Centralized Training System

This example demonstrates how to use the centralized training system for pneumonia detection from chest X-ray images.

## Quick Start

### 1. Prepare Your Dataset

Create a zip file with the following structure:
```
pneumonia_dataset.zip
├── metadata.csv          # CSV with patient IDs and labels
└── images/              # Directory containing X-ray images
    ├── patient001.png
    ├── patient002.png
    └── ...
```

**CSV Format:**
```csv
patientId,Target
patient001,0
patient002,1
patient003,0
```

- `patientId`: Unique identifier for each patient
- `Target`: Binary classification label (0=Normal, 1=Pneumonia)

### 2. Run Training

#### Simple Usage
```python
from federated_pneumonia_detection.src.control.centralized_trainer import CentralizedTrainer

# Initialize trainer
trainer = CentralizedTrainer(
    checkpoint_dir="my_checkpoints",
    logs_dir="my_logs"
)

# Train from zip file
results = trainer.train_from_zip(
    zip_path="path/to/pneumonia_dataset.zip",
    experiment_name="my_experiment"
)

print(f"Best model: {results['best_model_path']}")
```

#### Command Line Usage
```bash
# Basic training
python federated_pneumonia_detection/train_centralized.py dataset.zip

# With custom settings
python federated_pneumonia_detection/train_centralized.py dataset.zip \
    --experiment-name my_experiment \
    --checkpoint-dir ./checkpoints \
    --logs-dir ./logs

# Validate only (no training)
python federated_pneumonia_detection/train_centralized.py dataset.zip --validate-only
```

## Features

### Automatic Dataset Processing
- ✅ **Zip Extraction**: Automatically extracts and processes zip files
- ✅ **CSV Detection**: Auto-discovers CSV files in the zip
- ✅ **Image Validation**: Validates image files and formats
- ✅ **Train/Val Split**: Stratified splitting maintaining class balance

### Advanced Training
- ✅ **Class Weighting**: Automatic balancing for imbalanced datasets
- ✅ **Early Stopping**: Prevents overfitting with configurable patience
- ✅ **Model Checkpointing**: Saves best models based on validation metrics
- ✅ **Learning Rate Monitoring**: Tracks and logs LR changes
- ✅ **GPU Acceleration**: Automatic GPU detection and mixed precision

### Comprehensive Logging
- ✅ **TensorBoard Integration**: Real-time training visualization
- ✅ **Metric Tracking**: Accuracy, Precision, Recall, F1, AUC
- ✅ **Progress Monitoring**: Detailed logging throughout training
- ✅ **Model Summaries**: Complete model architecture information

## Configuration

### Default Configuration
The system uses sensible defaults for pneumonia detection:
- ResNet50 V2 backbone with ImageNet pretraining
- 224x224 input images with standard preprocessing
- Binary classification with BCE loss
- Adam optimizer with weight decay
- ReduceLROnPlateau scheduler

### Custom Configuration
Create a YAML configuration file:

```yaml
# config/my_experiment.yaml
experiment_config:
  max_epochs: 50
  learning_rate: 0.001
  batch_size: 32
  validation_split: 0.2
  early_stopping_patience: 7

system_constants:
  random_seed: 42
  image_size: 224
```

Use it in training:
```python
trainer = CentralizedTrainer(config_path="config/my_experiment.yaml")
```

## Output Structure

After training, you'll have:

```
project/
├── checkpoints/
│   ├── pneumonia_model_epoch=15_val_recall=0.850.ckpt  # Best model
│   ├── pneumonia_model_loss_epoch=12_val_loss=0.245.ckpt
│   └── last.ckpt  # Most recent checkpoint
└── logs/
    └── my_experiment/
        └── version_0/
            ├── events.out.tfevents.*  # TensorBoard logs
            └── hparams.yaml  # Hyperparameters
```

## Advanced Usage

### Custom Callbacks
```python
from pytorch_lightning.callbacks import ModelCheckpoint

# Custom checkpoint callback
custom_checkpoint = ModelCheckpoint(
    monitor='val_f1',
    mode='max',
    save_top_k=5
)

# Add to trainer (requires modifying callback setup)
```

### Model Evaluation
```python
# Load best model for evaluation
from federated_pneumonia_detection.src.control.lit_resnet import LitResNet

model = LitResNet.load_from_checkpoint(results['best_model_path'])
model.eval()

# Get predictions
predictions = model.predict_step(test_batch, 0)
```

### Monitoring Training
```bash
# Start TensorBoard
tensorboard --logdir=logs

# Navigate to http://localhost:6006
```

## Troubleshooting

### Common Issues

**1. Zip file validation fails**
- Ensure CSV has 'patientId' and 'Target' columns
- Check image files are named as {patientId}.png/.jpg
- Verify zip file is not corrupted

**2. Out of memory errors**
- Reduce batch_size in configuration
- Use gradient accumulation
- Enable gradient checkpointing

**3. Training doesn't improve**
- Check class balance in dataset
- Verify image quality and labels
- Adjust learning rate and patience

### Getting Help
- Check logs in the specified logs directory
- Enable DEBUG logging for detailed information
- Review the guidelines in `documentation/guidelines.md`

## Integration with Federated Learning

This centralized trainer serves as the foundation for federated learning:
- Same model architecture used in federated clients
- Training metrics compatible with federated aggregation
- Checkpoint format supports federated parameter sharing

See the federated learning documentation for distributed training setup.