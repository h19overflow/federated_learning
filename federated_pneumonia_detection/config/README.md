# Configuration Manager

This directory contains the centralized configuration management system for the Federated Pneumonia Detection project.

## Files

- `default_config.yaml` - Main configuration file with all default settings
- `config_manager.py` - ConfigManager class for reading/writing configuration
- `example_usage.py` - Example code showing how to use ConfigManager
- `__init__.py` - Module initialization and imports

## ConfigManager Class

The `ConfigManager` class provides a centralized way to read and modify any field in the YAML configuration file using dot notation.

### Basic Usage

```python
from config_manager import ConfigManager

# Create a config manager instance
config = ConfigManager()

# Read values using dot notation
learning_rate = config.get('experiment.learning_rate')
img_size = config.get('system.img_size')
batch_size = config.get('system.batch_size')

# Set values using dot notation
config.set('experiment.learning_rate', 0.002)
config.set('system.batch_size', 256)

# Save changes to file
config.save()
```

### Advanced Features

```python
# Dictionary-style access
learning_rate = config['experiment.learning_rate']
config['experiment.epochs'] = 25

# Check if key exists
if 'experiment.learning_rate' in config:
    print("Key exists")

# Get entire configuration sections
experiment_config = config.get_section('experiment')
system_config = config.get_section('system')

# Update multiple values at once
config.update({
    'experiment.learning_rate': 0.002,
    'system.batch_size': 256,
    'experiment.epochs': 20
})

# List all available keys
all_keys = config.list_keys()
experiment_keys = config.list_keys('experiment')

# Create backup before making changes
backup_path = config.backup()

# Reset to original state
config.reset()

# Reload from file
config.reload()
```

### Convenience Functions

For quick one-off operations:

```python
from config_manager import quick_get, quick_set

# Quick read
learning_rate = quick_get('experiment.learning_rate')

# Quick write (automatically saves)
quick_set('experiment.learning_rate', 0.002)
```

### Import from Package

```python
# Import from the config package
from config import ConfigManager, get_config_manager, quick_get, quick_set

# Or create via factory function
config = get_config_manager()
```

## Configuration Structure

The configuration is organized into sections:

- **system**: Image size, batch size, validation split, etc.
- **paths**: File and directory paths
- **columns**: Data column mappings
- **experiment**: Model training parameters, federated learning settings
- **output**: Output directory configurations
- **logging**: Logging configuration

### Example Configuration Keys

```
system.img_size                    # [256, 256]
system.batch_size                  # 512
experiment.learning_rate           # 0.0015
experiment.epochs                  # 15
experiment.num_clients            # 5
paths.base_path                   # "."
logging.level                     # "INFO"
```

## Error Handling

The ConfigManager includes robust error handling:

- `KeyError` for non-existent configuration keys
- `ValueError` for invalid nested value assignments  
- `FileNotFoundError` for missing configuration files

## Thread Safety

The ConfigManager is **not** thread-safe. If you need to modify configuration from multiple threads, implement appropriate locking mechanisms.

## Examples

Run the example file to see all features in action:

```bash
python example_usage.py
```

This will demonstrate reading, writing, and various ConfigManager features without modifying the actual configuration file.