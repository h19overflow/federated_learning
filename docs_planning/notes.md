# Notes: DIAGRAMS.md Research

## Source 1: system_architecture_diagram.md
Key findings:
- System has 4 main layers: Frontend, API/Orchestration, Execution/Data, Infrastructure
- Two training modes: Centralized (single computer) and Federated (multiple hospitals collaborating)
- Main flows: Centralized Training, Federated Training, Inference, Real-time Metrics
- Technologies: React/FastAPI, PyTorch Lightning, Flower framework, PostgreSQL

## System Components Extracted

### Frontend Layer
- Web Application (React + TypeScript)
- WebSocket Client (for live updates)

### API/Orchestration Layer
- API Gateway (FastAPI)
- Service Orchestrator (business logic coordinator)

### Execution/Data Layer
- Flower Framework (federated learning coordination)
- ML Engine (PyTorch Lightning - models, training)
- Data Access (SQLAlchemy - database operations)

### Infrastructure Layer
- File Storage (model checkpoints, results)
- WebSocket Relay (broadcasts live metrics)
- PostgreSQL + pgvector (database)
- Weights & Biases (experiment tracking)

## Key Data Flows Identified

### 1. Centralized Training
- User starts training → Data loaded → Model trains → Metrics collected → Results saved → Live updates sent

### 2. Federated Training
- User starts federated training → Server coordinates → Multiple hospitals train locally → Updates combined → Results saved

### 3. Inference (Diagnosis)
- User uploads X-ray → Image validated → Model analyzes → Result returned with heatmap

### 4. Real-time Metrics
- Training loop produces metrics → WebSocket relay broadcasts → Frontend displays live charts

## Technical Terms to Simplify
- "Federated Learning" → "Collaborative training without sharing patient data"
- "Inference" → "Making predictions/diagnoses"
- "Model checkpoints" → "Saved versions of the trained AI"
- "Embeddings" → "Mathematical representations of data"
- "gRPC" → "Communication protocol"
- "ResNet50" → "Type of AI model architecture"
