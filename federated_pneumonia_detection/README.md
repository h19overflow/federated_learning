# Federated Pneumonia Detection System

**A comprehensive federated learning platform for distributed chest X-ray pneumonia classification using PyTorch, Flower, and FastAPI.**

---

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Module Documentation](#module-documentation)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)

---

## System Overview

This system implements both **centralized** and **federated** learning approaches for pneumonia detection:

### Centralized Training
- Single-machine training on full dataset
- Fast iteration for development and baseline models
- Complete data available for evaluation
- Reference: [src/control/dl_model/README.md](src/control/dl_model/README.md)

### Federated Learning
- Distributed training across multiple clients
- Data stays on client machines (privacy-preserving)
- Server coordinates aggregation via Flower framework
- Server-side evaluation on held-out test set
- Reference: [src/control/federated_new_version/README.md](src/control/federated_new_version/README.md)

---

## Architecture

### Clean Architecture Layers

```
┌──────────────────────────────────────────────────────────────┐
│ API Layer (FastAPI)                                          │
│ - REST endpoints (/experiments, /runs, /configuration)      │
│ - WebSocket (ws://localhost:8765) for real-time metrics     │
│ - Request validation & error handling                        │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│ Boundary Layer (Data Access)                                 │
│ - Database: PostgreSQL with SQLAlchemy ORM                   │
│ - CRUD operations: Run, Client, RunMetric, ServerEvaluation │
│ - External services: WandB, Vector DB integration            │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│ Control Layer (Business Logic)                               │
│ - Centralized: CentralizedTrainer orchestration              │
│ - Federated: Server/Client apps, aggregation strategy       │
│ - Metrics: Collection, persistence, real-time streaming     │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│ Entities Layer (Domain Models)                               │
│ - ResNetWithCustomHead: ResNet50 + custom binary head        │
│ - CustomImageDataset: PyTorch dataset for X-ray loading      │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│ Utils Layer (Shared Helpers)                                 │
│ - Data processing: CSV loading, train/val splitting          │
│ - Image transforms: Augmentation, preprocessing pipelines    │
│ - Logging: Structured logging infrastructure                 │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow: End-to-End

```
═══════════════════════════════════════════════════════════════
CENTRALIZED TRAINING
═══════════════════════════════════════════════════════════════

User uploads dataset.zip
         ↓
FastAPI: POST /experiments/centralized
         ↓
DataSourceExtractor: Extract and validate
         ↓
load_metadata() → DataFrame with filenames & labels
         ↓
create_train_val_split() → 80% train, 20% validation
         ↓
XRayDataModule: Create PyTorch DataLoaders
         ↓
LitResNet: PyTorch Lightning model
         ↓
Training Loop (per epoch):
  ├─ Forward pass (ResNet50 + custom head)
  ├─ Loss computation (BCEWithLogitsLoss)
  ├─ Backward pass (AdamW optimizer)
  ├─ Metrics update (Accuracy, Precision, Recall, F1, AUROC)
  ├─ Validation evaluation
  └─ Callback chain:
     ├─ ModelCheckpoint: Save best by val_recall
     ├─ EarlyStopping: Stop if no improvement
     ├─ MetricsCollector: Extract and stream metrics
     └─ WebSocketSender: Broadcast to frontend
         ↓
Real-Time Frontend Updates (TrainingExecution component)
         ↓
Training Complete
         ↓
MetricsCollector: Persist to database (run_metrics)
         ↓
API: GET /runs/{run_id}/metrics
         ↓
Frontend: ResultsVisualization component displays:
  ├─ Training curves (loss, accuracy over epochs)
  ├─ Metric cards (best accuracy, precision, recall, F1)
  ├─ Confusion matrix (2x2 grid)
  └─ Download options (CSV, JSON, summary)

═══════════════════════════════════════════════════════════════
FEDERATED LEARNING
═══════════════════════════════════════════════════════════════

User initiates federated training
         ↓
FastAPI: POST /experiments/federated
         ↓
Flower Server: Initialize global model
         ↓
Data Partitioner: Split dataset → Partition 0, 1, ..., K
         ↓
For each ROUND (1 to num_rounds):
  │
  ├─1. SERVER → CLIENTS: Send global model weights + config
  │
  ├─2. CLIENTS (parallel): Local training on partitions
  │    ├─ Load global model from server
  │    ├─ Train on local partition (max_epochs)
  │    ├─ Compute metrics: train_loss, train_acc, etc.
  │    └─ Return: Updated weights + metrics + num-examples
  │
  ├─3. SERVER: Aggregate weights (FedAvg)
  │    └─ Weighted average: Σ(weight_i × num_examples_i)
  │
  ├─4. CLIENTS: Evaluate global model
  │    └─ Return: test_loss, test_accuracy, etc.
  │
  ├─5. SERVER: Aggregate evaluation metrics
  │    └─ Weighted by num-examples per client
  │
  ├─6. SERVER: Centralized evaluation on test set
  │    ├─ Forward pass on server's held-out data
  │    ├─ Compute: loss, accuracy, precision, recall, f1, auroc
  │    ├─ Extract: confusion matrix (TP, TN, FP, FN)
  │    └─ Persist to ServerEvaluation table
  │
  ├─7. DATABASE: Store metrics
  │    ├─ run_metrics: Per-client, per-round training metrics
  │    ├─ server_evaluations: Global model performance
  │    └─ run_metrics: Aggregated client evaluation metrics
  │
  ├─8. WEBSOCKET: Broadcast round metrics to frontend
  │
  └─Continue to next round
         ↓
All rounds complete
         ↓
API: GET /runs/{run_id}/server-evaluation
         ↓
Frontend: Display per-round metrics & trends
```

---

## Key Features

### 🔒 Privacy-Preserving Federated Learning
- Data never leaves client machines
- Only model weights transmitted
- Supports heterogeneous (non-IID) data distributions

### ⚡ Real-Time Monitoring
- WebSocket-based live metric streaming
- Training progress dashboard
- Per-round metrics for federated learning

### 📊 Comprehensive Metrics
- Classification metrics: Accuracy, Precision, Recall, F1, AUROC
- Confusion matrix (True Positives, False Positives, etc.)
- Per-epoch training history
- Per-round aggregated statistics

### 🔧 Production-Ready
- FastAPI with automatic API documentation
- PostgreSQL persistence with SQLAlchemy ORM
- Configuration management (YAML-based)
- Structured logging with error handling
- PyTorch Lightning for reproducible training

### 🧪 Flexible Experimentation
- Centralized baseline for comparison
- Configurable federated parameters (rounds, clients, epochs)
- Hyperparameter overrides per experiment
- Model checkpointing and early stopping

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd federated_pneumonia_detection

# Install dependencies
uv install
```

### Run Centralized Training

```bash
# Via API (recommended)
uvicorn src.api.main:app --reload --port 8001
# Then POST to http://localhost:8001/experiments/centralized

# Or directly via Python
python -m federated_pneumonia_detection.src.control.dl_model.centralized_trainer
```

### Run Federated Learning

```bash
# Via Flower simulation
uv run flwr run federated_pneumonia_detection/src/control/federated_new_version

# Or via PowerShell
./federated_pneumonia_detection/src/rf.ps1
```

### Access Dashboard

```
Frontend: http://localhost:3000 (separate React app)
API Docs: http://localhost:8001/docs
WebSocket: ws://localhost:8765
```

---

## Module Documentation

| Module | Purpose | Documentation |
|--------|---------|---------------|
| **src/entities/** | Domain models (neural network, dataset) | [README.md](src/entities/README.md) |
| **src/utils/** | Shared utilities (data loading, transforms, logging) | [README.md](src/utils/README.md) |
| **src/boundary/** | Data access layer (database CRUD) | [README.md](src/boundary/README.md) |
| **src/control/dl_model/utils/** | Training utilities (Lightning, metrics, callbacks) | [README.md](src/control/dl_model/utils/README.md) |
| **src/control/dl_model/** | Centralized training orchestration | [README.md](src/control/dl_model/README.md) |
| **src/control/federated_new_version/** | Federated learning (Flower) | [README.md](src/control/federated_new_version/README.md) |
| **src/api/** | REST API endpoints and WebSocket | [README.md](src/api/README.md) |
| **config/** | Configuration management | [README.md](config/README.md) |

---

## Data Flow Diagram

### Training Pipeline

```
┌─────────────────────┐
│   Dataset (ZIP)     │
│  - metadata.csv     │
│  - Images/ dir      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ DataSourceExtractor                 │
│ - Extract ZIP                       │
│ - Validate structure                │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Data Processing (utils)             │
│ - Load CSV                          │
│ - Split train/val                   │
│ - Create DataLoaders                │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Model & Training                    │
│ - ResNetWithCustomHead              │
│ - LitResNet wrapper                 │
│ - Training loop                     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Metrics Collection                  │
│ - MetricsCollector callback         │
│ - Per-epoch: loss, accuracy, etc.   │
│ - Confusion matrix tracking         │
└──────────┬──────────────────────────┘
           │
           ├─────────────────┬────────────────┐
           │                 │                │
           ▼                 ▼                ▼
    ┌────────────┐   ┌────────────┐   ┌────────────┐
    │ WebSocket  │   │ Database   │   │ JSON/CSV   │
    │ (Frontend) │   │ (Persist)  │   │ (Export)   │
    └────────────┘   └────────────┘   └────────────┘
           │                 │                │
           └─────────────────┼────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ API Endpoints    │
                    │ /runs/{run_id}   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Frontend Viz     │
                    │ - Charts         │
                    │ - Metrics cards  │
                    │ - Confusion mtx  │
                    └──────────────────┘
```

---

## Technology Stack

### Deep Learning
- **PyTorch 2.8.0**: Neural network framework
- **PyTorch Lightning 2.5.5**: Training orchestration
- **torchvision 0.23.0**: Vision utilities (ResNet, transforms)
- **torchmetrics**: Metric computation (Accuracy, Precision, etc.)

### Federated Learning
- **Flower[simulation] 1.22.0**: Federated learning framework
- **Flower-datasets[vision] 0.5.0**: Vision datasets

### Web & API
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI web server
- **WebSockets**: Real-time communication

### Data & Storage
- **PostgreSQL**: Relational database
- **SQLAlchemy 2.x**: ORM for database operations
- **Pydantic**: Data validation and schemas
- **pandas**: Data manipulation

### Configuration & Utilities
- **PyYAML**: Configuration files
- **python-dotenv**: Environment variable management
- **scikit-learn**: ML utilities (train_test_split, class_weight)
- **PIL/Pillow**: Image loading
- **numpy**: Numerical operations

### AI/ML Tools
- **LangChain**: Agentic systems framework
- **MCP (Model Context Protocol)**: Tool calling
- **arXiv API**: Research paper integration

---

## Project Structure

```
federated_pneumonia_detection/
├── __init__.py
├── requirements.txt              # Dependencies
├── config/
│   ├── default_config.yaml      # Configuration file
│   ├── config_manager.py        # Configuration access
│   └── README.md                # Config documentation
│
├── src/
│   ├── api/                     # FastAPI endpoints
│   │   ├── main.py             # Entry point
│   │   └── endpoints/          # Organized by resource
│   │
│   ├── boundary/                # Data access layer
│   │   ├── engine.py           # Database models
│   │   └── CRUD/               # CRUD operations
│   │
│   ├── control/                 # Business logic
│   │   ├── dl_model/           # Centralized training
│   │   └── federated_new_version/ # Federated learning
│   │
│   ├── entities/                # Domain models
│   │   ├── resnet_with_custom_head.py
│   │   └── custom_image_dataset.py
│   │
│   └── utils/                   # Shared utilities
│       ├── data_processing.py
│       ├── image_transforms.py
│       └── loggers/
```

---

## Metrics & Evaluation

### Tracked Metrics
- **Loss**: Binary cross-entropy loss
- **Accuracy**: Correct predictions / total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **AUROC**: Area under ROC curve
- **Confusion Matrix**: TP, TN, FP, FN components

### Model Checkpoints
- Saved based on validation recall (best model)
- Kept: Top 3 + last checkpoint
- Format: `.ckpt` (PyTorch Lightning)

---

## Contributing

- Use type hints throughout
- Include error handling for I/O operations
- Add structured logging
- One file = one responsibility (max 150 lines)

---

## Support & Documentation

- **API Documentation**: Visit `/docs` after starting API server
- **Module READMEs**: Each module has comprehensive documentation
- **Configuration**: See [config/README.md](config/README.md)
- **Code Examples**: See specific module documentation

---

## Related Repositories

- **Frontend**: [xray-vision-ai-forge](../xray-vision-ai-forge/) - React dashboard
- **Models**: Trained models stored in PostgreSQL

---

## License

[Add your license here]

---

**Last Updated**: 2024-12-22

For issues, questions, or contributions, please contact the development team.
