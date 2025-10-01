# Federated Pneumonia Detection System

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange.svg)](https://pytorchlightning.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A sophisticated federated learning system for pneumonia detection from chest X-ray images, built with PyTorch Lightning and designed for privacy-preserving collaborative medical AI.

## 🎯 Project Overview

This Final Year Project (FYP) implements a federated learning framework specifically designed for pneumonia detection, enabling multiple medical institutions to collaboratively train AI models while keeping sensitive patient data local and private.

### Key Features

- **🔒 Privacy-Preserving**: Federated learning ensures patient data never leaves local institutions
- **🏥 Medical AI**: Specialized for pneumonia detection from chest X-rays
- **⚡ Modern Architecture**: Built on PyTorch Lightning for scalable, production-ready training
- **🔧 Modular Design**: Clean architecture with separation of concerns
- **📊 Comprehensive Evaluation**: Advanced metrics and visualization tools
- **🧪 Well-Tested**: Extensive unit and integration test coverage

## 🏗️ Architecture

```
federated_pneumonia_detection/
├── src/
│   ├── boundary/           # External interfaces and APIs
│   ├── control/            # Business logic and orchestration
│   │   ├── dl_model/       # Deep learning model management
│   │   └── reporting/      # Evaluation and visualization
│   ├── entities/           # Core domain objects
│   └── utils/              # Shared utilities
├── models/                 # Configuration and constants
├── config/                 # Configuration files
├── data/                   # Dataset management
├── examples/               # Usage examples
├── tests/                  # Test suites
└── logs/                   # Training and execution logs
```

### Core Components

1. **Federated Learning Engine**: Implements FedAvg and advanced FL strategies
2. **Data Pipeline**: Efficient image processing with custom transformations
3. **Model Architecture**: ResNet-based architecture with custom heads
4. **Evaluation System**: Comprehensive metrics including AUC, F1, sensitivity
5. **Visualization Tools**: Training curves, confusion matrices, and model analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- At least 8GB RAM
- 10GB+ storage for datasets

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FYP2
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Verify installation**
   ```bash
   python main.py
   ```

### Dataset Setup

1. **Prepare your chest X-ray dataset**
   - Place images in `Training/` or `Test/` directories
   - Ensure metadata CSV files contain required columns:
     - `patient_id`: Unique patient identifier
     - `filename`: Image filename
     - `label`: Binary classification (0=Normal, 1=Pneumonia)

2. **Configure paths**
   - Update `federated_pneumonia_detection/config/` with your dataset paths
   - Modify `SystemConstants` if needed for your data structure

## 💡 Usage Examples

### Basic Data Pipeline

```python
from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.src.utils.data_processing import load_and_split_data
from federated_pneumonia_detection.src.control.dl_model.utils.model import XRayDataModule

# Load configuration
config_loader = ConfigLoader()
constants = config_loader.create_system_constants()
config = config_loader.create_experiment_config()

# Prepare data
train_df, val_df = load_and_split_data(constants, config)

# Create data module
data_module = XRayDataModule(
    train_df=train_df,
    val_df=val_df,
    constants=constants,
    config=config,
    image_dir="path/to/images"
)
```

### Centralized Training

```python
from federated_pneumonia_detection.src.control.dl_model import CentralizedTrainer

# Initialize trainer
trainer = CentralizedTrainer(
    constants=constants,
    config=config,
    logger=logger
)

# Train model
results = trainer.train_from_csv(
    csv_path="metadata.csv",
    image_dir="images/"
)

print(f"Training complete! Best validation accuracy: {results['best_val_acc']:.3f}")
```

### Model Evaluation

```python
from federated_pneumonia_detection.src.control.reporting import ModelEvaluator

# Evaluate model
evaluator = ModelEvaluator(constants, config)
metrics = evaluator.evaluate_model(
    model=trained_model,
    test_loader=test_dataloader
)

print(f"Test Metrics:")
print(f"  AUC: {metrics['auc']:.3f}")
print(f"  F1 Score: {metrics['f1']:.3f}")
print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
print(f"  Specificity: {metrics['specificity']:.3f}")
```

### Complete Pipeline Example

Run the complete pipeline demonstration:

```bash
python federated_pneumonia_detection/examples/complete_data_pipeline_example.py
```

## ⚙️ Configuration

### System Constants

Configure dataset paths and processing parameters in `SystemConstants`:

```python
class SystemConstants:
    BASE_PATH: str = "data/"
    METADATA_FILENAME: str = "train_metadata.csv"
    IMG_SIZE: Tuple[int, int] = (224, 224)
    PATIENT_ID_COLUMN: str = "patient_id"
    TARGET_COLUMN: str = "label"
    FILENAME_COLUMN: str = "filename"
```

### Experiment Configuration

Customize training parameters in `ExperimentConfig`:

```python
config = ExperimentConfig(
    learning_rate=0.001,
    epochs=50,
    batch_size=32,
    num_clients=5,          # For federated learning
    num_rounds=10,          # FL communication rounds
    validation_split=0.2,
    augmentation_strength=1.0
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=federated_pneumonia_detection

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
```

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end data pipelines
- **Fixtures**: Reusable test data and configurations in `conftest.py`

## 📊 Model Performance

### Metrics Tracked

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Medical AI Metrics**: Sensitivity, Specificity, AUC-ROC
- **Training Metrics**: Loss curves, Learning rate schedules
- **Federated Learning**: Communication rounds, client contributions

### Visualization Tools

The system includes comprehensive visualization capabilities:

- Training/validation curves
- Confusion matrices
- ROC curves and AUC analysis
- Feature importance maps
- Client contribution tracking (FL)

## 🔮 Federated Learning Features

### Supported Strategies

- **FedAvg**: Standard federated averaging
- **Advanced Strategies**: Custom aggregation methods
- **Client Selection**: Smart client sampling
- **Privacy Mechanisms**: Differential privacy support

### Multi-Institution Simulation

```python
# Simulate multiple medical institutions
from federated_pneumonia_detection.src.control import AdvancedTrainingStrategies

fl_trainer = AdvancedTrainingStrategies(
    num_clients=5,
    num_rounds=20,
    client_fraction=0.6
)

results = fl_trainer.federated_training(
    client_datasets=client_data
)
```

## 🛠️ Development

### Project Structure Philosophy

This project follows **Clean Architecture** principles:

- **Entities**: Core business logic and domain models
- **Use Cases**: Application-specific business rules
- **Interface Adapters**: Controllers, presenters, and gateways
- **Frameworks & Drivers**: External concerns (PyTorch, file I/O)

### Adding New Features

1. **Models**: Add new configurations to `models/`
2. **Data Processing**: Extend utilities in `src/utils/`
3. **Training Logic**: Implement in `src/control/`
4. **Tests**: Always add corresponding tests
5. **Documentation**: Update README and docstrings

### Code Quality

- **Type Hints**: Full type annotation coverage
- **Docstrings**: Comprehensive documentation
- **Testing**: >90% code coverage target
- **Linting**: Follow PEP 8 standards

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Lightning** for the excellent deep learning framework
- **Medical AI Community** for advancing healthcare through AI
- **Federated Learning Research** for privacy-preserving ML innovations

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out through the project's GitHub page.

---

**Note**: This is a research project for educational purposes. Always consult with medical professionals and follow proper regulatory guidelines when deploying medical AI systems in clinical settings.