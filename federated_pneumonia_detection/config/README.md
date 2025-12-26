# Configuration System ⚙️

**Centralized configuration management for the Federated Pneumonia Detection System.**

This module handles all application settings, allowing for flexible configuration via YAML files and environment variable overrides without changing code.

---

## 📂 Configuration Structure

The configuration is managed by the `ConfigManager` class, which loads defaults from `default_config.yaml` and allows runtime overrides.

### Files
- **`default_config.yaml`**: The single source of truth for all default parameters.
- **`config_manager.py`**: Python class for type-safe access and modification.

---

## 🔑 Key Configuration Sections

### 1. System (`system`)
Global parameters affecting both centralized and federated modes.
```yaml
system:
  img_size: [256, 256]      # Input image dimensions
  batch_size: 512           # Training batch size
  validation_split: 0.2     # Fraction of data for validation
  seed: 42                  # Random seed for reproducibility
```

### 2. Experiment (`experiment`)
Hyperparameters for model training.
```yaml
experiment:
  learning_rate: 0.0015
  epochs: 15                # Max epochs (Centralized)
  num_rounds: 15            # Max rounds (Federated)
  dropout_rate: 0.3         # Model regularization
  architecture: "resnet50"  # Backbone model
```

### 3. Paths (`paths`)
Directory locations for data and outputs.
```yaml
paths:
  base_path: "."
  main_images_folder: "Images"
```

### 4. Output (`output`)
Where results and logs are stored.
```yaml
output:
  checkpoint_dir: "models/checkpoints"
  results_dir: "results"
  log_dir: "logs"
```

---

## 💻 Usage

### Accessing Config in Code
```python
from federated_pneumonia_detection.config.config_manager import ConfigManager

config = ConfigManager()

# Get a value (dot notation)
lr = config.get("experiment.learning_rate")

# Set a value
config.set("experiment.epochs", 50)
```

### Overriding via Environment Variables
Any key can be overridden using the `FPD_` prefix + uppercase key name (nested keys use double underscores).

- **Override Batch Size**: `FPD_SYSTEM__BATCH_SIZE=64`
- **Override Learning Rate**: `FPD_EXPERIMENT__LEARNING_RATE=0.005`
