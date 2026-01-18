# Boundary Layer

Data access layer connecting business logic to PostgreSQL, ML models, and vector databases.

## What It Does

```mermaid
flowchart LR
    subgraph Control["Control Layer"]
        Train[Training]
        Chat[Chat Agent]
        Infer[Inference]
    end

    subgraph Boundary["Boundary Layer"]
        CRUD[CRUD Classes]
        InfSvc[Inference Service]
        VDB[Vector DB Engine]
    end

    subgraph External["External Systems"]
        DB[(PostgreSQL)]
        Model[ML Model]
        Embed[HuggingFace Embeddings]
    end

    Train --> CRUD --> DB
    Infer --> InfSvc --> Model
    Chat --> VDB --> Embed
    Chat --> CRUD
```

## Entity Relationships

```mermaid
erDiagram
    RUN ||--o{ CLIENT : "has"
    RUN ||--o{ RUN_METRIC : "tracks"
    RUN ||--o{ SERVER_EVALUATION : "evaluates"
    CLIENT ||--o{ ROUND : "participates"

    RUN {
        int id PK
        string training_mode "centralized|federated"
        string status "in_progress|completed|failed"
        timestamp start_time
    }

    RUN_METRIC {
        int id PK
        int run_id FK
        int client_id FK "NULL for centralized"
        string metric_name
        float metric_value
        int step
    }

    SERVER_EVALUATION {
        int id PK
        int run_id FK
        int round_number
        float accuracy
        float recall
        json confusion_matrix
    }
```

## Module Overview

| Module | Purpose | Pattern |
|--------|---------|---------|
| **engine.py** | DB connection + session factory | Singleton, connection pooling |
| **models/** | SQLAlchemy ORM definitions | Declarative base |
| **CRUD/** | Data access operations | Generic base class |
| **inference_service.py** | ML prediction wrapper | Lazy singleton |
| **vdb_query_engine.py** | Semantic search for RAG | PGVector + HuggingFace |

## Database Engine

Connection pooling for production reliability:

```mermaid
flowchart LR
    App[Application] --> Pool[Connection Pool]
    Pool --> C1[Conn 1]
    Pool --> C2[Conn 2]
    Pool --> C3[Conn ...]
    Pool --> C5[Conn 5]
    C1 --> DB[(PostgreSQL)]
    C2 --> DB
    C3 --> DB
    C5 --> DB

    style Pool fill:#f39c12,stroke:#d35400,color:#fff
```

| Setting | Value |
|---------|-------|
| Pool size | 5 |
| Max overflow | 10 |
| Pre-ping | Enabled |
| Recycle | 1 hour |

## CRUD Pattern

All CRUD classes inherit from a generic base:

```mermaid
classDiagram
    class BaseCRUD~T~ {
        +create(obj)
        +get(id)
        +update(id, data)
        +delete(id)
        +count()
        +exists(id)
        +bulk_create(items)
    }

    BaseCRUD <|-- RunCRUD
    BaseCRUD <|-- ClientCRUD
    BaseCRUD <|-- RoundCRUD
    BaseCRUD <|-- RunMetricCRUD
    BaseCRUD <|-- ServerEvaluationCRUD

    class RunCRUD {
        +get_by_status()
        +persist_metrics()
    }

    class RunMetricCRUD {
        +get_by_metric_name()
        +get_best_metric()
        +metric_stats()
    }

    class ServerEvaluationCRUD {
        +get_summary_stats()
        +get_best_by_metric()
    }
```

## Data Flow

### Centralized Training
```mermaid
sequenceDiagram
    participant T as Trainer
    participant C as RunCRUD
    participant DB as PostgreSQL

    T->>C: persist_metrics(epoch=1)
    C->>DB: INSERT run_metric
    Note over DB: client_id = NULL
```

### Federated Training
```mermaid
sequenceDiagram
    participant S as FL Server
    participant C as CRUD
    participant DB as PostgreSQL

    S->>C: create_client("client_0")
    S->>C: create_round(round=1)

    loop Each Client
        S->>C: persist_metrics(client_id, round_id)
    end

    S->>C: create_evaluation(round=1)
    C->>DB: INSERT server_evaluation
    Note over DB: Full confusion matrix + AUROC
```

## ML Services

### Inference Service

Wraps model prediction with clinical interpretation:

```mermaid
flowchart LR
    Image[X-ray Image] --> Infer[InferenceEngine]
    Infer --> Pred[Prediction]
    Pred --> Clinical[Clinical Agent]
    Clinical --> Result[Interpreted Result]

    style Clinical fill:#27ae60,stroke:#1e8449,color:#fff
```

### Vector DB Engine

Semantic search for research papers:

```mermaid
flowchart LR
    Query[User Query] --> Embed[all-MiniLM-L6-v2]
    Embed --> Vector[Query Vector]
    Vector --> PG[(PGVector)]
    PG --> Results[Top 15 Papers]

    style Embed fill:#3498db,stroke:#2980b9,color:#fff
```

## Key Files

```
boundary/
├── engine.py              # DB connection, session factory
├── inference_service.py   # ML prediction wrapper
├── vdb_query_engine.py    # Semantic search (RAG)
├── cleanup_database.py    # DB maintenance utilities
├── models/                # ORM entities
│   ├── run.py
│   ├── client.py
│   ├── round.py
│   ├── run_metric.py
│   ├── server_evaluation.py
│   └── chat_session.py
└── CRUD/                  # Data access classes
    ├── base.py            # Generic CRUD template
    ├── run.py
    ├── run_metric.py
    ├── server_evaluation.py
    ├── client.py
    ├── round.py
    ├── chat_history.py
    └── fetch_documents.py
```

## Quick Reference

| Action | Class/Method |
|--------|--------------|
| Create training run | `RunCRUD.create()` |
| Save epoch metrics | `RunCRUD.persist_metrics()` |
| Get best accuracy | `ServerEvaluationCRUD.get_best_by_metric('accuracy')` |
| Run prediction | `InferenceService.predict(image)` |
| Search papers | `VDBQueryEngine.query("federated learning")` |
| List chat sessions | `get_all_chat_sessions()` |
