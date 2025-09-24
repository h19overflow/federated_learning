# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a federated pneumonia detection system that combines centralized and federated learning approaches for chest X-ray image classification. The system uses PyTorch, Flower (for federated learning), FastAPI, and Streamlit to create a complete ML pipeline with web dashboard.

## Architecture

### Three-Layer Architecture (Entity-Control-Boundary)
- **Entities** (`src/entities/`): Data structures and configuration classes
  - `SystemConstants`: Global configuration values with immutable defaults
  - `ExperimentConfig`: Experiment parameters with validation
  - `CustomImageDataset`, `ResNetWithCustomHead`, `SessionResultsCache`: Core ML entities
- **Control** (`src/control/`): Business logic and orchestration
- **Boundary** (`src/boundary/`): External interfaces (FastAPI, Streamlit, file I/O)
- **Utils** (`src/utils/`): Support functions and data processing

### Configuration System
The system uses a layered configuration approach:
- YAML files in `config/` directory with `default_config.yaml` as base
- `ConfigLoader` class handles loading and environment variable overrides
- Environment variables prefixed with `FPD_` override config values
- Configuration is validated through entity classes

### Data Processing Pipeline
- Metadata loading from CSV with patient IDs and targets
- Stratified sampling to maintain class balance
- Configurable train/validation splits
- Image preprocessing (224x224, ImageNet normalization)
- Path validation and error handling

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv pip install -r requirements.txt

# Run with Python module syntax
python -m federated_pneumonia_detection.main
```

### Code Quality
```bash
# Formatting
black federated_pneumonia_detection/

# Linting
flake8 federated_pneumonia_detection/

# Type checking
mypy federated_pneumonia_detection/

# Testing
pytest tests/ --cov=federated_pneumonia_detection
pytest tests/test_specific_module.py -v
```

### Development Workflow
The project follows a phase-based development approach (see `documentation/guidelines.md`):
1. **Phase 1**: Foundation & Entity classes (Weeks 1-2)
2. **Phase 2**: Core ML components (Weeks 3-4)
3. **Phase 3**: Centralized training (Weeks 5-6)
4. **Phase 4**: Dashboard foundation (Weeks 7-8)
5. **Phase 5**: Federated learning (Weeks 9-10)
6. **Phase 6**: Advanced dashboard (Weeks 11-12)
7. **Phase 7**: Integration & testing (Week 13)
8. **Phase 8**: Final polish (Week 14)

## Key Patterns

### Configuration Pattern
```python
# Load configuration
config_loader = ConfigLoader()
constants = config_loader.create_system_constants()
exp_config = config_loader.create_experiment_config()

# Use in components
data_processor = DataProcessor(constants)
train_df, val_df = data_processor.load_and_process_data(exp_config)
```

### Error Handling Pattern
All classes implement comprehensive error handling with logging:
- Input validation in constructors
- Try-catch blocks with specific exceptions
- Detailed error messages for debugging
- Graceful fallbacks where appropriate

### Reproducibility Pattern
- All random operations use configurable seeds
- SystemConstants provides consistent defaults
- Configuration validation prevents invalid parameter combinations

## Data Structure
- Expected directory structure: `Images/Images/` for image files
- Metadata CSV with `patientId` and `Target` columns
- Images named as `{patientId}.png`
- Configurable paths and file patterns through SystemConstants

## ML Components (Planned)
- ResNet50 V2 backbone with custom binary classification head
- PyTorch Lightning modules for training orchestration
- Flower integration for federated learning
- Comprehensive metrics (accuracy, precision, recall, F1, AUC)
- Model checkpointing and experiment tracking

## Development Notes
- Use absolute paths from project root
- Prefer constructor initialization over function-level object creation
- Follow SOLID principles with dependency injection
- Maintain files under 150 lines maximum
- Use type hints throughout
- Document all public methods and classes