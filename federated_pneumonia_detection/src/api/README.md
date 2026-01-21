# API Layer

Backend server connecting the React frontend to ML training, inference, and research capabilities.

## What It Does

```mermaid
flowchart LR
    subgraph Frontend
        UI[React App]
    end

    subgraph API["API Layer"]
        direction TB
        Main[FastAPI Server]
        MW[Security Middleware]
    end

    subgraph Features
        Train[Training]
        Infer[Inference]
        Chat[Research Chat]
        Runs[Results]
    end

    subgraph Backend
        DB[(PostgreSQL)]
        WS[WebSocket]
        Model[ML Models]
    end

    UI -->|HTTP/WS| Main
    Main --> MW --> Features
    Features --> DB
    Features --> WS
    Features --> Model
    WS -->|Live Metrics| UI
```

## Architecture

The API is organized into **9 modular routers**, each handling a specific domain:

```mermaid
graph TD
    subgraph Core["Core (main.py)"]
        App[FastAPI App]
        Life[Startup Services]
        DI[Dependency Injection]
    end

    subgraph Routers["Endpoint Routers"]
        R1["/config" - Settings]
        R2["/experiments/centralized"]
        R3["/experiments/federated"]
        R4["/experiments/status"]
        R5["/runs" - Analytics]
        R6["/chat" - Research]
        R7["/inference" - Predictions]
        R8["/reports" - PDF Export]
        R9["/stream" - SSE Events]
    end

    App --> Life
    App --> DI
    DI --> Routers

    classDef core fill:#4A90D9,stroke:#2E5A8C,color:#fff
    classDef router fill:#7CB342,stroke:#558B2F,color:#fff
    class App,Life,DI core
    class R1,R2,R3,R4,R5,R6,R7,R8,R9 router
```

## Module Overview

| Module              | Purpose                | Key Patterns                              |
| ------------------- | ---------------------- | ----------------------------------------- |
| **experiments/**    | Start training jobs    | Background tasks, subprocess spawning     |
| **inference/**      | Run predictions        | Single/batch processing, GradCAM heatmaps |
| **runs_endpoints/** | Query training results | Pluggable exporters, strategy pattern     |
| **chat/**           | Research assistant     | SSE streaming, ArXiv RAG integration      |
| **streaming/**      | Real-time updates      | WebSocket broadcast, SSE polling          |
| **reports/**        | Generate PDFs          | Template rendering                        |
| **middleware/**     | Security               | Prompt injection detection                |

## Data Flow

### Training Flow

```mermaid
sequenceDiagram
    participant UI as React UI
    participant API as FastAPI
    participant BG as Background Task
    participant WS as WebSocket
    participant DB as Database

    UI->>API: POST /experiments/centralized
    API-->>UI: 202 Accepted + experiment_id
    API->>BG: Queue training task

    loop Each Epoch
        BG->>WS: Broadcast metrics
        WS-->>UI: Live update
        BG->>DB: Save metrics
    end

    BG->>DB: Mark complete
    UI->>API: GET /runs/{id}/metrics
    API-->>UI: Full results
```

### Inference Flow

```mermaid
sequenceDiagram
    participant UI as React UI
    participant API as FastAPI
    participant Model as ML Model

    UI->>API: POST /inference/predict
    Note over API: Validate image
    API->>Model: Run prediction
    Model-->>API: Confidence + class
    API-->>UI: Result + interpretation
```

## Key Files

```
api/
├── main.py           # App bootstrap, router mounting, CORS
├── deps.py           # Dependency injection (singletons)
├── settings.py       # Environment config
├── middleware/       # Security (prompt injection detection)
├── services/         # Startup initialization
└── endpoints/
    ├── experiments/  # Training orchestration
    ├── inference/    # Model predictions
    ├── runs_endpoints/  # Results & analytics
    ├── chat/         # Research assistant
    ├── reports/      # PDF generation
    ├── streaming/    # WebSocket + SSE
    └── schema/       # Pydantic models
```

## Real-Time Communication

Two channels for live updates:

| Channel       | Port | Use Case                 |
| ------------- | ---- | ------------------------ |
| **SSE**       | 8001 | Training progress events |
| **WebSocket** | 8765 | Metric broadcasts        |

## Security

- **CORS**: Allows localhost:5173 (Vite dev) and :8080
- **Prompt Injection Detection**: Blocks malicious chat queries
  - 5 attack categories (jailbreak, exfiltration, role-hijacking, etc.)
  - 10KB query limit, 70% repetition threshold

## Startup Sequence

```mermaid
flowchart LR
    A[Database] --> B[WebSocket Server]
    B --> C[MCP Manager]
    C --> D[W&B Tracker]

    style A fill:#e74c3c,stroke:#c0392b,color:#fff
    style B fill:#f39c12,stroke:#d35400,color:#fff
    style C fill:#3498db,stroke:#2980b9,color:#fff
    style D fill:#27ae60,stroke:#1e8449,color:#fff
```

Database is **critical** (fails startup). Others are optional with warnings.

## Quick Reference

| Action                     | Endpoint                              |
| -------------------------- | ------------------------------------- |
| Start centralized training | `POST /experiments/centralized/train` |
| Start federated training   | `POST /experiments/federated/train`   |
| Check training status      | `GET /experiments/status/{id}`        |
| Get run metrics            | `GET /runs/{id}/metrics`              |
| Download results           | `GET /runs/{id}/download/csv`         |
| Run single prediction      | `POST /inference/predict`             |
| Run batch prediction       | `POST /inference/predict-batch`       |
| Generate heatmap           | `POST /inference/heatmap`             |
| Query research chat        | `POST /chat/query/stream`             |
