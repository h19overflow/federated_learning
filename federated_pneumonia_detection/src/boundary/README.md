# Boundary Layer

**Date**: 2026-01-24
**Agent**: Sub-Apollo (Documentation Specialist)

## Problem
The Boundary layer required clear documentation explaining its role as the data access layer between business logic and external systems. The existing documentation needed better visual representation of database relationships and CRUD patterns.

## Solution
Created comprehensive documentation with professional Mermaid diagrams showing the database schema, CRUD patterns, and service integrations. The documentation clearly explains the Entity-Control-Boundary pattern implementation.

### Key Implementation Files
- `src/boundary/engine.py:1` - Database connection and session factory
- `src/boundary/models/base.py:1` - SQLAlchemy declarative base
- `src/boundary/CRUD/base.py:1` - Generic CRUD base class
- `src/boundary/vdb_query_engine.py:1` - Vector database query engine for RAG

### System Connections

```mermaid
flowchart TB
    subgraph Control["Control Layer"]
        Train[Training Orchestrator]
        Chat[Research Chat Agent]
        Infer[Inference Engine]
        Analytics[Analytics Service]
    end

    subgraph Boundary["Boundary Layer"]
        direction TB
        CRUD[CRUD Classes]
        Engine[Database Engine]
        VDB[Vector DB Engine]
        InfSvc[Inference Service]
    end

    subgraph External["External Systems"]
        DB[(PostgreSQL)]
        Model[ML Models]
        Embed[HuggingFace Embeddings]
        Files[File System]
    end

    Train --> CRUD
    Chat --> VDB
    Infer --> InfSvc
    Analytics --> CRUD

    CRUD --> Engine
    Engine --> DB
    VDB --> Embed
    InfSvc --> Model
    CRUD --> Files

    classDef boundary fill:#7ED321,stroke:#5FA818,color:#fff
    classDef control fill:#BD10E0,stroke:#8B0A50,color:#fff
    classDef external fill:#CCCCCC,stroke:#999999,color:#000

    class CRUD,Engine,VDB,InfSvc boundary
    class Train,Chat,Infer,Analytics control
    class DB,Model,Embed,Files external
```

### Database Schema

```mermaid
erDiagram
    RUN ||--o{ CLIENT : "has"
    RUN ||--o{ RUN_METRIC : "tracks"
    RUN ||--o{ SERVER_EVALUATION : "evaluates"
    CLIENT ||--o{ ROUND : "participates"
    ROUND ||--o{ RUN_METRIC : "generates"
    CHAT_SESSION ||--o{ CHAT_HISTORY : "contains"

    RUN {
        int id PK
        string training_mode "centralized|federated"
        string status "in_progress|completed|failed"
        timestamp start_time
        timestamp end_time
        string run_name
        json config_snapshot
    }

    CLIENT {
        int id PK
        int run_id FK
        string client_id "client_0, client_1..."
        int num_examples
        json partition_info
        timestamp created_at
    }

    ROUND {
        int id PK
        int run_id FK
        int round_number
        timestamp start_time
        timestamp end_time
        int num_clients_participated
    }

    RUN_METRIC {
        int id PK
        int run_id FK
        int client_id FK "NULL for centralized"
        int round_id FK "NULL for centralized"
        string metric_name
        float metric_value
        int step "epoch or round"
        timestamp recorded_at
    }

    SERVER_EVALUATION {
        int id PK
        int run_id FK
        int round_number
        float accuracy
        float precision
        float recall
        float f1_score
        float auroc
        json confusion_matrix
        timestamp evaluated_at
    }

    CHAT_SESSION {
        int id PK
        string session_id
        timestamp created_at
        timestamp last_activity
        json metadata
    }

    CHAT_HISTORY {
        int id PK
        int session_id FK
        string role "user|assistant|system"
        text content
        json sources
        timestamp timestamp
    }
```

### CRUD Pattern Implementation

```mermaid
classDiagram
    class BaseCRUD~T~ {
        <<abstract>>
        +db: Session
        +model: Type[T]
        +create(obj_in: CreateSchemaType) T
        +get(id: int) Optional[T]
        +get_multi(skip: int, limit: int) List[T]
        +update(id: int, obj_in: UpdateSchemaType) T
        +delete(id: int) T
        +count() int
        +exists(id: int) bool
        +bulk_create(objects: List[CreateSchemaType]) List[T]
    }

    class RunCRUD {
        +get_by_status(status: str) List[Run]
        +persist_metrics(run_id: int, metrics: Dict) None
        +get_with_metrics(id: int) Optional[RunWithMetrics]
        +update_status(id: int, status: str) Run
    }

    class ClientCRUD {
        +get_by_run_id(run_id: int) List[Client]
        +create_client(run_id: int, client_id: str) Client
        +get_client_metrics(client_id: int) List[RunMetric]
    }

    class RunMetricCRUD {
        +get_by_metric_name(metric_name: str) List[RunMetric]
        +get_best_metric(run_id: int, metric_name: str) Optional[RunMetric]
        +metric_stats(run_id: int, metric_name: str) MetricStats
        +get_metrics_by_round(run_id: int, round_num: int) List[RunMetric]
    }

    class ServerEvaluationCRUD {
        +get_by_round(run_id: int, round_num: int) Optional[ServerEvaluation]
        +get_summary_stats(run_id: int) SummaryStats
        +get_best_by_metric(run_id: int, metric_name: str) Optional[ServerEvaluation]
    }

    BaseCRUD <|-- RunCRUD
    BaseCRUD <|-- ClientCRUD
    BaseCRUD <|-- RoundCRUD
    BaseCRUD <|-- RunMetricCRUD
    BaseCRUD <|-- ServerEvaluationCRUD
    BaseCRUD <|-- ChatSessionCRUD
    BaseCRUD <|-- ChatHistoryCRUD

    classDef base fill:#4A90E2,stroke:#357ABD,color:#fff
    classDef concrete fill:#7ED321,stroke:#5FA818,color:#fff

    class BaseCRUD base
    class RunCRUD,ClientCRUD,RoundCRUD,RunMetricCRUD,ServerEvaluationCRUD,ChatSessionCRUD,ChatHistoryCRUD concrete
```

### Database Engine Configuration

```mermaid
flowchart LR
    App[Application] --> Pool[Connection Pool]
    
    subgraph Pool["Connection Pool (5 connections)"]
        C1[Connection 1]
        C2[Connection 2]
        C3[Connection 3]
        C4[Connection 4]
        C5[Connection 5]
    end

    subgraph Overflow["Overflow (10 max)"]
        O1[Overflow 1]
        O2[Overflow 2]
        O3[Overflow ...]
    end

    Pool --> DB[(PostgreSQL)]
    Overflow --> DB

    DB --> PrePing[Pre-ping Check]
    PrePing --> Recycle[Recycle after 1 hour]

    classDef pool fill:#F5A623,stroke:#D58512,color:#fff
    classDef db fill:#7ED321,stroke:#5FA818,color:#fff
    classDef config fill:#4A90E2,stroke:#357ABD,color:#fff

    class Pool,C1,C2,C3,C4,C5,O1,O2,O3 pool
    class DB,PrePing,Recycle db
    class App config
```

**Connection Pool Settings:**
- **Pool Size**: 5 connections
- **Max Overflow**: 10 additional connections
- **Pre-ping**: Enabled (validates connections before use)
- **Recycle**: 1 hour (prevents stale connections)

### Data Flow Patterns

#### Centralized Training Data Flow

```mermaid
sequenceDiagram
    participant Trainer as CentralizedTrainer
    participant RunDAO as RunCRUD
    participant MetricDAO as RunMetricCRUD
    participant DB as PostgreSQL

    Trainer->>RunDAO: create(run_data)
    RunDAO->>DB: INSERT INTO run
    DB-->>RunDAO: run_id
    RunDAO-->>Trainer: Run object

    loop Each Epoch
        Trainer->>MetricDAO: persist_metrics(run_id, epoch_metrics)
        Note over MetricDAO: client_id = NULL for centralized
        MetricDAO->>DB: INSERT INTO run_metric
        DB-->>MetricDAO: Success
    end

    Trainer->>RunDAO: update_status(run_id, "completed")
    RunDAO->>DB: UPDATE run SET status
```

#### Federated Learning Data Flow

```mermaid
sequenceDiagram
    participant Server as FL Server
    participant RunDAO as RunCRUD
    participant ClientDAO as ClientCRUD
    participant RoundDAO as RoundCRUD
    participant MetricDAO as RunMetricCRUD
    participant EvalDAO as ServerEvaluationCRUD
    participant DB as PostgreSQL

    Server->>RunDAO: create(run_data)
    Server->>ClientDAO: create_client(run_id, "client_0")
    Server->>ClientDAO: create_client(run_id, "client_1")

    loop Each Round
        Server->>RoundDAO: create_round(run_id, round_num)
        
        par Client Metrics
            Server->>MetricDAO: persist_metrics(client_id, round_id, metrics)
        and Server Evaluation
            Server->>EvalDAO: create_evaluation(run_id, round_num, eval_metrics)
        end
    end

    Server->>RunDAO: update_status(run_id, "completed")
```

### Vector Database Integration

```mermaid
flowchart LR
    Query[User Query] --> Embed[all-MiniLM-L6-v2]
    Embed --> Vector[Query Vector]
    
    subgraph PGVector["PGVector (PostgreSQL Extension)"]
        Search[<--> Search]
        Index[IVFFlat Index]
    end

    Vector --> Search
    Search --> Index
    Index --> Results[Top 15 Documents]
    
    Results --> Context[Context Assembly]
    Context --> LLM[LLM Generation]

    classDef embedding fill:#BD10E0,stroke:#8B0A50,color:#fff
    classDef vector fill:#7ED321,stroke:#5FA818,color:#fff
    classDef llm fill:#F5A623,stroke:#D58512,color:#fff

    class Embed,Vector embedding
    class PGVector,Search,Index,Results vector
    class Context,LLM llm
```

### Inference Service Architecture

```mermaid
flowchart LR
    Image[X-ray Image] --> Preprocess[Preprocessing]
    Preprocess --> Model[ResNet50 Model]
    Model --> Predict[Prediction]
    
    Predict --> Clinical[Clinical Agent]
    Predict --> GradCAM[GradCAM Service]
    
    Clinical --> Risk[Risk Assessment]
    GradCAM --> Heatmap[Attention Heatmap]
    
    Risk --> Result[Final Result]
    Heatmap --> Result

    classDef inference fill:#4A90E2,stroke:#357ABD,color:#fff
    classDef clinical fill:#BD10E0,stroke:#8B0A50,color:#fff
    classDef output fill:#7ED321,stroke:#5FA818,color:#fff

    class Image,Preprocess,Model,Predict inference
    class Clinical,Risk clinical
    class GradCAM,Heatmap,Result output
```

### Decision Rationale
- **SQLAlchemy Chosen**: Powerful ORM with connection pooling and migrations
- **Generic CRUD Base**: Reduces code duplication with type-safe operations
- **PGVector Integration**: Native PostgreSQL vector search for RAG
- **Connection Pooling**: Ensures production reliability and performance

### Integration Points
- **Upstream**: Called by Control layer for all data operations
- **Downstream**: Manages PostgreSQL database and file system
- **External**: Integrates with HuggingFace for embeddings
- **Models**: Uses SQLAlchemy models for ORM mapping

## Key Files Structure

```
src/boundary/
├── engine.py                    # Database connection and session factory
├── inference_service.py         # ML prediction wrapper
├── vdb_query_engine.py          # Vector database search (RAG)
├── cleanup_database.py          # Database maintenance utilities
├── models/                       # SQLAlchemy ORM definitions
│   ├── base.py                  # Declarative base and mixins
│   ├── run.py                   # Training run entity
│   ├── client.py                # FL client entity
│   ├── round.py                 # FL round entity
│   ├── run_metric.py            # Metrics entity
│   ├── server_evaluation.py     # FL server evaluation
│   ├── chat_session.py          # Chat session entity
│   └── chat_history.py          # Chat message history
└── CRUD/                         # Data access operations
    ├── base.py                  # Generic CRUD base class
    ├── run.py                   # Run CRUD operations
    ├── run_metric.py            # Metric CRUD operations
    ├── server_evaluation.py     # Evaluation CRUD
    ├── client.py                # Client CRUD
    ├── round.py                 # Round CRUD
    ├── chat_history.py          # Chat CRUD
    └── fetch_documents.py       # Document retrieval for RAG
```

## Quick Reference

| Operation | Class/Method | Description |
|-----------|--------------|-------------|
| Create training run | `RunCRUD.create()` | Initialize new training run |
| Save epoch metrics | `RunCRUD.persist_metrics()` | Store training metrics |
| Get best accuracy | `ServerEvaluationCRUD.get_best_by_metric('accuracy')` | Find best performing model |
| Run prediction | `InferenceService.predict(image)` | ML model inference |
| Search papers | `VDBQueryEngine.query("federated learning")` | Semantic document search |
| List chat sessions | `ChatSessionCRUD.get_multi()` | Get all chat sessions |
| Create FL client | `ClientCRUD.create_client()` | Register federated client |
| Get round metrics | `RunMetricCRUD.get_metrics_by_round()` | Fetch FL round data |

## Database Operations

### Connection Management

```python
# Session dependency pattern
from sqlalchemy.orm import Session

def get_db() -> Generator[Session, None, None]:
    """Database session dependency with automatic cleanup"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Transaction Patterns

```python
# Atomic operations
def create_run_with_metrics(db: Session, run_data: dict, metrics: list):
    """Create run and metrics in single transaction"""
    try:
        run = RunCRUD(db).create(run_data)
        for metric in metrics:
            RunMetricCRUD(db).create({
                "run_id": run.id,
                **metric
            })
        db.commit()
        return run
    except Exception:
        db.rollback()
        raise
```

## Performance Optimizations

- **Connection Pooling**: 5 base connections + 10 overflow
- **Query Optimization**: Indexed foreign keys and metric names
- **Batch Operations**: Bulk inserts for metrics
- **Lazy Loading**: Relationships loaded on demand
- **Connection Recycling**: 1-hour timeout prevents stale connections

## Error Handling

- **Database Errors**: Wrapped in domain-specific exceptions
- **Connection Failures**: Automatic retry with exponential backoff
- **Validation Errors**: Pydantic schemas for data integrity
- **Constraint Violations**: Clear error messages for duplicate entries