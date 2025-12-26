# FastAPI Layer - Training Orchestration & Results Management

**Purpose**: RESTful API interface providing training orchestration, real-time metrics streaming, and results retrieval for the federated pneumonia detection system.

---

## Table of Contents

1. [Overview](#overview)
2. [Functional Architecture](#functional-architecture)
3. [Request/Response Flows](#requestresponse-flows)
4. [Endpoint Groups](#endpoint-groups)
5. [Integration Flow](#integration-flow)
6. [Component Details](#component-details)

---

## Overview

**Framework**: FastAPI (async Python web framework)
**Server**: Uvicorn ASGI
**Base URL**: `http://localhost:8001`
**WebSocket**: `ws://localhost:8765` (real-time metrics relay)

**Key Capabilities**:
- **Orchestration**: Start centralized or federated training runs.
- **Persistence**: Query results and metrics from PostgreSQL.
- **Monitoring**: Stream real-time progress via WebSocket.
- **Assistant**: Query Arxiv and local RAG via the chat endpoints.

---

## Backend Architecture (ECB)

```mermaid
graph TB
    subgraph Boundary ["🧱 Boundary Layer (Interfaces)"]
        direction TB
        API_End["API Endpoints"]
        Run_DAO["Run DAO"]
        Metric_DAO["Metric DAO"]
    end

    subgraph Control ["🎮 Control Layer (Logic)"]
        direction TB
        CT["Centralized Trainer"]
        FL_Server["FL ServerApp"]
        FL_Client["FL ClientApp"]
    end

    subgraph Entities ["📦 Entities Layer (Data)"]
        direction TB
        ResNet["ResNet Model"]
        Dataset["Custom Dataset"]
        Config["Config Manager"]
    end

    %% Flow
    API_End -->|Invokes| CT
    API_End -->|Invokes| FL_Server
    
    CT -->|Uses| ResNet
    CT -->|Uses| Dataset
    CT -->|Uses| Config
    
    FL_Server -->|Uses| ResNet
    FL_Client -->|Uses| ResNet
    
    CT -->|Persists via| Run_DAO
    CT -->|Persists via| Metric_DAO

    %% Styling
    classDef boundary fill:#AA00FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef control fill:#2962FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef entities fill:#D50000,stroke:#fff,stroke-width:2px,color:#fff;

    class API_End,Run_DAO,Metric_DAO boundary;
    class CT,FL_Server,FL_Client control;
    class ResNet,Dataset,Config entities;
```

---

## Request/Response Flows

### Centralized Training Sequence

```mermaid
sequenceDiagram
    autonumber
    participant API as FastAPI
    participant CT as CentralizedTrainer
    participant Model as ResNet Model
    participant DB as Database
    participant WS as WebSocket

    Note over API, CT: Initialization
    API->>CT: Start Training (Config)
    CT->>CT: Load Data & Split
    CT->>Model: Initialize Weights

    Note over CT, Model: Training Loop
    loop Every Epoch
        CT->>Model: Forward/Backward Pass
        Model-->>CT: Loss & Metrics
        CT->>WS: Broadcast Metrics
        CT->>DB: Save Metric Record
    end

    Note over CT, DB: Completion
    CT->>DB: Update Run Status (Done)
    CT->>WS: Broadcast "Training Complete"
```

### Flow 2: Federated Learning (Complete Lifecycle)

```mermaid
sequenceDiagram
    participant UI as React UI
    participant API as FastAPI
    participant Task as Background Task
    participant Server as Flower ServerApp
    participant Clients as ClientApp ×N
    participant Send as WebSocket Sender
    participant WS as WebSocket Server
    participant DB as PostgreSQL

    UI->>API: POST /experiments/federated<br/>{data_zip, num_rounds=15, num_clients=5}
    API-->>UI: 202 Accepted<br/>{experiment_id}

    Task->>Task: Extract data<br/>Create partitions
    Task->>Server: Initialize Flower

    Server->>Server: Load global model<br/>Create run record
    Server->>DB: INSERT run record

    Server->>Send: send_training_mode(True, 15, 5)
    Send->>WS: JSON {type: training_mode}
    WS->>UI: WebSocket update
    UI->>UI: Update → "FL Mode: 15 rounds, 5 clients"

    loop For each Round (1 to num_rounds)
        Server->>Server: Get global weights
        Server->>Clients: Send weights + config

        par Client 0
            Clients->>Clients: Load partition 0
            Clients->>Clients: Train locally
            Clients->>Clients: Compute metrics
        and Client 1
            Clients->>Clients: Load partition 1
            Clients->>Clients: Train locally
            Clients->>Clients: Compute metrics
        and Client N
            Clients->>Clients: Load partition N
            Clients->>Clients: Train locally
            Clients->>Clients: Compute metrics
        end

        Clients-->>Server: Return weights + metrics

        Server->>Server: Aggregate (FedAvg)
        Server->>Server: Evaluate on server test set
        Server->>DB: Persist metrics & evaluations

        Server->>Send: send_round_metrics(round, metrics)
        Send->>WS: JSON {type: round_metrics}
        WS->>UI: WebSocket broadcast
        UI->>UI: Update round progress
    end

    Server->>DB: Mark run completed
    Server->>Send: send_training_end(run_id)
    Send->>WS: JSON {type: training_end}
    WS->>UI: Final signal

    UI->>API: GET /api/runs/42/federated-rounds
    API->>DB: SELECT * FROM server_evaluations
    DB-->>API: Per-round metrics
    API-->>UI: {rounds, metrics_per_round}
    UI->>UI: Render federated results

    style UI fill:#e1f5ff
    style API fill:#fff3e0
    style Server fill:#c8e6c9
    style DB fill:#e8f5e9
```

### Flow 3: Results Retrieval

```mermaid
sequenceDiagram
    participant UI as React UI
    participant API as FastAPI
    participant Cache as Optional Cache
    participant DB as PostgreSQL
    participant Transform as Data Transform

    UI->>API: GET /api/runs/42/metrics

    API->>API: Validate run_id

    alt Cache exists
        API->>Cache: Check cached result
        Cache-->>API: Return cached
    else
        API->>DB: SELECT * FROM runs WHERE id=42
        API->>DB: SELECT * FROM run_metrics WHERE run_id=42
        DB-->>API: [Run record, Metrics array]
    end

    API->>Transform: Transform to ExperimentResults
    Transform->>Transform: Organize by phase<br/>Compute aggregates<br/>Format for frontend

    API-->>UI: 200 OK<br/>{training_history, best_metrics, confusion_matrix}

    UI->>UI: Parse response<br/>Update state<br/>Render charts

```

---

## Endpoint Groups

### Group 1: Training Orchestration (`/experiments`)

| Endpoint | Method | Purpose | Status Code |
|----------|--------|---------|------------|
| `/experiments/centralized` | POST | Start centralized training | 202 |
| `/experiments/federated` | POST | Start federated training | 202 |
| `/experiments/status/{exp_id}` | GET | Poll training status | 200 |
| `/experiments/list` | GET | List all experiments | 200 |

**See**: [endpoints/experiments/](endpoints/experiments/) for implementation

---

### Group 2: Results Management (`/api/runs`)

| Endpoint | Method | Purpose | Status Code |
|----------|--------|---------|------------|
| `/api/runs/list` | GET | List all runs | 200 |
| `/api/runs/{run_id}/metrics` | GET | Get training metrics | 200 |
| `/api/runs/{run_id}/federated-rounds` | GET | Get per-round metrics (FL only) | 200 |
| `/api/runs/{run_id}/server-evaluation` | GET | Get server evaluations (FL only) | 200 |
| `/api/runs/{run_id}/download/json` | GET | Download as JSON | 200 |
| `/api/runs/{run_id}/download/csv` | GET | Download as CSV | 200 |
| `/api/runs/{run_id}/download/summary` | GET | Download text summary | 200 |

**See**: [endpoints/runs_endpoints/](endpoints/runs_endpoints/) for implementation

---

### Group 3: Configuration (`/config`)

| Endpoint | Method | Purpose | Status Code |
|----------|--------|---------|------------|
| `/config/current` | GET | Get current config | 200 |
| `/config/update` | POST | Update config | 200 |

**See**: [endpoints/configuration_settings/](endpoints/configuration_settings/) for implementation

---

### Group 4: Chat & RAG (`/chat`)

| Endpoint | Method | Purpose | Status Code |
|----------|--------|---------|------------|
| `/chat/query` | POST | Query with RAG + Arxiv | 200 |
| `/chat/query/stream` | POST | Stream response (SSE) | 200 |
| `/chat/history/{session_id}` | GET | Get conversation history | 200 |
| `/chat/history/{session_id}` | DELETE | Clear history | 200 |

**See**: [endpoints/chat/](endpoints/chat/) for implementation

---

## Integration Flow

### How the API Connects Everything

```mermaid
graph LR
    subgraph External["External"]
        React["React Frontend"]
        Browser["Browser<br/>WebSocket"]
    end

    subgraph API_Layer["API Layer"]
        MainApp["main.py<br/>FastAPI App"]
        Router1["experiments<br/>Router"]
        Router2["runs<br/>Router"]
        Router3["config<br/>Router"]
        Router4["chat<br/>Router"]
        DI_Layer["deps.py<br/>Dependency Injection"]
    end

    subgraph Services["Services"]
        TaskQ["Background<br/>Task Queue"]
        Config["ConfigManager"]
        CRUD["CRUD Operations"]
    end

    subgraph Backend["Backend Execution"]
        Trainer["Training<br/>Components"]
        Metrics["Metrics<br/>Sender"]
    end

    subgraph DataLayer["Data Persistence"]
        Database[(PostgreSQL)]
        WebSocketSrv["WebSocket<br/>Server"]
    end

    React -->|HTTP POST| MainApp
    React -->|HTTP GET| MainApp
    Browser -->|WebSocket| WebSocketSrv

    MainApp -->|Route| Router1
    MainApp -->|Route| Router2
    MainApp -->|Route| Router3
    MainApp -->|Route| Router4

    Router1 -->|Get deps| DI_Layer
    Router2 -->|Get deps| DI_Layer
    Router3 -->|Get deps| DI_Layer
    Router4 -->|Get deps| DI_Layer

    DI_Layer -->|Provide| Config
    DI_Layer -->|Provide| CRUD

    Router1 -->|Queue| TaskQ
    Router2 -->|Query| CRUD
    Router3 -->|R/W| Config
    Router4 -->|Q/Search| CRUD

    TaskQ -->|Initialize| Trainer
    Trainer -->|Report| Metrics
    Metrics -->|Send| WebSocketSrv
    WebSocketSrv -->|Broadcast| Browser

    Trainer -->|Save| Database
    CRUD -->|Query| Database

    %% Styling
    classDef ext fill:#007BFF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef api fill:#FF6F00,stroke:#fff,stroke-width:2px,color:#fff;
    classDef srv fill:#AA00FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef exe fill:#2962FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef dat fill:#00C853,stroke:#fff,stroke-width:2px,color:#fff;

    class External,React,Browser ext;
    class API_Layer,MainApp,Router1,Router2,Router3,Router4,DI_Layer api;
    class Services,TaskQ,Config,CRUD srv;
    class Backend,Trainer,Metrics exe;
    class DataLayer,Database,WebSocketSrv dat;
```

---

## Component Details

### 1. Main Application (`main.py`)

**Responsibilities**:
- FastAPI app initialization with async lifespan management
- CORS configuration for frontend access
- Route registration
- WebSocket server startup

---

### 2. Dependency Injection (`deps.py`)

**Providers**:
- `get_db()`: SQLAlchemy session
- `get_config()`: ConfigManager for YAML config
- `get_experiment_crud()`: Run CRUD operations
- `get_run_metric_crud()`: Metric CRUD operations

---

### 3. Endpoint Routers

**Location**: `endpoints/` directory

**Organization**:
```
endpoints/
├── experiments/
│   ├── centralized_endpoints.py    # POST /experiments/centralized
│   ├── federated_endpoints.py      # POST /experiments/federated
│   ├── status_endpoints.py         # GET /experiments/status
│   └── utils/
│       ├── centralized_tasks.py    # Background task for centralized
│       ├── federated_tasks.py      # Background task for federated
│       └── file_handling.py        # ZIP extraction, validation
├── runs_endpoints/
│   ├── runs_list.py                # GET /api/runs/list
│   ├── runs_metrics.py             # GET /api/runs/{id}/metrics
│   ├── runs_federated_rounds.py    # GET /api/runs/{id}/federated-rounds
│   ├── runs_server_evaluation.py   # GET /api/runs/{id}/server-evaluation
│   ├── runs_download.py            # GET /api/runs/{id}/download/{format}
│   └── utils.py                    # Shared utilities
├── configuration_settings/
│   ├── configuration_endpoints.py  # GET/POST /config
│   └── schemas.py                  # Pydantic schemas
└── chat/
    ├── chat_endpoints.py           # POST /chat/query
    └── chat_utils.py               # RAG, Arxiv integration
```

---

### 4. Background Task Execution

**Flow**:
1. Endpoint validates request → returns 202 Accepted immediately
2. Task queued to background executor
3. Training starts asynchronously
4. Metrics sent to WebSocket in real-time
5. Results persisted to database
6. Frontend polls `/api/runs/{id}/metrics` for results

---

## Error Handling Strategy

```mermaid
graph TD
    A["Request arrives<br/>at endpoint"] -->|Valid| B["Execute endpoint logic"]
    A -->|Invalid| C["Validation error<br/>422 Unprocessable"]

    B -->|Success| D["Return 200/202"]
    B -->|Resource not found| E["404 Not Found"]
    B -->|Database error| F["500 Internal Server Error"]
    B -->|Training error| G["Log error<br/>Send via WebSocket<br/>Update status"]

    D -->|Response| H["Frontend receives<br/>result"]
    C -->|Response| H
    E -->|Response| H
    F -->|Response| H
    G -->|WebSocket| H

    %% Styling
    classDef start fill:#6200EA,stroke:#fff,stroke-width:2px,color:#fff;
    classDef success fill:#00C853,stroke:#fff,stroke-width:2px,color:#fff;
    classDef error fill:#D50000,stroke:#fff,stroke-width:2px,color:#fff;
    classDef endpt fill:#0091EA,stroke:#fff,stroke-width:2px,color:#fff;

    class A start;
    class B,D success;
    class C,E,F,G error;
    class H endpt;
```

---

## Related Documentation

- **WebSocket Metrics**: [dl_model/utils/data/README.md](../control/dl_model/utils/data/README.md) - Real-time metric streaming
- **Federated Learning**: [control/federated_new_version/README.md](../control/federated_new_version/README.md) - FL orchestration
- **Centralized Training**: [control/dl_model/README.md](../control/dl_model/README.md) - Centralized training logic
- **System Architecture**: [README.md](../../README.md) - Overall system design
- **Dependency Injection**: [deps.py](deps.py) - Service providers
- **Configuration**: [settings.py](settings.py) - API configuration

