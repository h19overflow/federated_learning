# Control Layer

Business logic hub orchestrating ML training, federated learning, and AI research assistants.

## What It Does

```mermaid
flowchart TB
    subgraph Control["Control Layer"]
        direction TB
        CT[Centralized Training]
        FL[Federated Learning]
        Agent[Research Agents]
        Infer[Inference Engine]
    end

    API[API Layer] --> Control
    Control --> Boundary[Boundary Layer]
    Control --> WS[WebSocket]

    CT --> Model[ML Models]
    FL --> Model
    Infer --> Model
    Agent --> LLM[Claude/Gemini]
```

## Architecture

Four subsystems handle different concerns:

```mermaid
graph LR
    subgraph CentralTraining["dl_model/"]
        CT[CentralizedTrainer]
        CB[Callbacks]
        DM[XRayDataModule]
    end

    subgraph Federated["federated_new_version/"]
        SA[ServerApp]
        CA[ClientApp]
        Strat[FedAvg Strategy]
    end

    subgraph Agents["agentic_systems/"]
        Router[Query Router]
        ArXiv[ArXiv Agent]
        Clinical[Clinical Agent]
    end

    subgraph Inference["model_inferance/"]
        IE[Inference Engine]
        GC[GradCAM]
        PDF[Report Generator]
    end

    classDef training fill:#2962FF,stroke:#1565C0,color:#fff
    classDef agents fill:#7B1FA2,stroke:#4A148C,color:#fff
    classDef inference fill:#00897B,stroke:#004D40,color:#fff

    class CT,CB,DM,SA,CA,Strat training
    class Router,ArXiv,Clinical agents
    class IE,GC,PDF inference
```

## Module Overview

| Module                     | Purpose                           | Key Pattern                   |
| -------------------------- | --------------------------------- | ----------------------------- |
| **dl_model/**              | Single-node GPU training          | PyTorch Lightning + callbacks |
| **federated_new_version/** | Multi-client distributed training | Flower framework              |
| **agentic_systems/**       | LLM research assistants           | LangChain + MCP tools         |
| **model_inferance/**       | Predictions + explanations        | GradCAM heatmaps              |
| **report_generation/**     | PDF export                        | ReportLab templates           |

## Centralized Training Flow

```mermaid
sequenceDiagram
    participant API
    participant Trainer as CentralizedTrainer
    participant Model as LitResNet
    participant WS as WebSocket
    participant DB as Database

    API->>Trainer: train(config)
    Trainer->>Trainer: Extract ZIP + prepare data
    Trainer->>Model: Initialize ResNet50

    loop Each Epoch
        Model->>Model: Forward + backward pass
        Model-->>Trainer: Metrics (loss, accuracy, recall)
        Trainer->>WS: Broadcast live metrics
        Trainer->>DB: Persist to run_metrics
    end

    Trainer->>DB: Mark run completed
```

**Callbacks Chain:**

```mermaid
flowchart LR
    A[ModelCheckpoint] --> B[EarlyStopping]
    B --> C[MetricsCollector]
    C --> D[WebSocketSender]
    D --> E[GradientMonitor]

    style A fill:#e74c3c,stroke:#c0392b,color:#fff
    style C fill:#3498db,stroke:#2980b9,color:#fff
```

- **ModelCheckpoint**: Save top-3 by val_recall
- **EarlyStopping**: Patience=7 on val_recall
- **MetricsCollector**: CSV + JSON per epoch

## Federated Learning Flow

```mermaid
sequenceDiagram
    participant Server as FL Server
    participant C1 as Client 0
    participant C2 as Client 1
    participant DB as Database

    Server->>Server: Initialize global model
    Server->>DB: Create run (in_progress)

    loop Each Round
        Server->>C1: Send weights + config
        Server->>C2: Send weights + config

        par Local Training
            C1->>C1: Train on partition
            C2->>C2: Train on partition
        end

        C1-->>Server: Updated weights + num_examples
        C2-->>Server: Updated weights + num_examples

        Server->>Server: FedAvg aggregation
        Server->>Server: Evaluate on test set
        Server->>DB: Save server_evaluation
    end

    Server->>DB: Mark run completed
```

**Key Components:**
| Component | Role |
|-----------|------|
| **ServerApp** | Orchestrates rounds, broadcasts weights |
| **ClientApp** | Local training on data partition |
| **ConfigurableFedAvg** | Weighted aggregation by num_examples |
| **Partitioner** | IID/Non-IID data distribution |

## Research Agent Flow

```mermaid
flowchart LR
    Query[User Query] --> Router{Query Router}

    Router -->|"research"| Agent[ArXiv Agent]
    Router -->|"basic"| Direct[Direct LLM]

    Agent --> Tools{Tool Selection}
    Tools -->|papers| ArXiv[ArXiv Search]
    Tools -->|local| RAG[RAG Tool]

    ArXiv --> MCP[MCP Server]
    RAG --> VDB[(Vector DB)]

    Agent --> Stream[SSE Stream]
    Direct --> Stream

    style Router fill:#f39c12,stroke:#d35400,color:#fff
    style Agent fill:#9b59b6,stroke:#8e44ad,color:#fff
```

**Agents:**
| Agent | Purpose | Model |
|-------|---------|-------|
| **Query Router** | Classify research vs basic | Gemini 2.0 Flash |
| **ArXiv Agent** | Paper search + synthesis | Claude 3.5 Sonnet |
| **Clinical Agent** | Risk assessment from predictions | Claude 3.5 Sonnet |

## Inference Pipeline

```mermaid
flowchart LR
    Image[X-ray Image] --> Preprocess[Resize 224x224<br/>ImageNet Normalize]
    Preprocess --> Model[ResNet50]
    Model --> Sigmoid[Sigmoid]
    Sigmoid --> Class[Normal/Pneumonia]

    Model --> GradCAM[GradCAM]
    GradCAM --> Heatmap[Attention Heatmap]

    Class --> Clinical[Clinical Agent]
    Clinical --> Risk[Risk Assessment]

    Class --> PDF[PDF Report]
    Heatmap --> PDF
    Risk --> PDF

    style GradCAM fill:#e74c3c,stroke:#c0392b,color:#fff
    style Clinical fill:#9b59b6,stroke:#8e44ad,color:#fff
```

**GradCAM Process:**

1. Hook final conv layer
2. Backward pass on target class
3. Global average pool gradients
4. ReLU activation map
5. Overlay with 40% alpha

## Key Files

```
control/
├── dl_model/
│   ├── centralized_trainer.py      # Main training orchestrator
│   └── utils/
│       └── model/
│           ├── lit_resnet.py       # PyTorch Lightning model
│           ├── training_callbacks.py
│           └── xray_data_module.py
│
├── federated_new_version/
│   └── core/
│       ├── server_app.py           # FL server
│       ├── client_app.py           # FL client
│       └── custom_strategy.py      # FedAvg variant
│
├── agentic_systems/
│   └── multi_agent_systems/
│       └── chat/
│           ├── arxiv_agent/engine.py
│           ├── query_router.py
│           └── mcp_manager.py
│
├── model_inferance/
│   ├── inference_engine.py
│   └── gradcam.py
│
└── report_generation/
    └── pdf_report.py
```

## Metrics Pipeline

All training modes share a common metrics flow:

```mermaid
flowchart LR
    Train[Training Loop] --> Collector[MetricsCollector]

    Collector --> File[CSV/JSON Files]
    Collector --> WS[WebSocket Broadcast]
    Collector --> DB[(PostgreSQL)]
    Collector --> WandB[W&B Tracking]

    style Collector fill:#3498db,stroke:#2980b9,color:#fff
```

| Metric           | Centralized         | Federated               |
| ---------------- | ------------------- | ----------------------- |
| Loss             | Per epoch           | Per round (server eval) |
| Accuracy         | Per epoch           | Per round               |
| Recall           | Per epoch (primary) | Per round               |
| AUROC            | Per epoch           | Per round               |
| Confusion Matrix | Final               | Per round               |

## Quick Reference

| Action                     | Entry Point                          |
| -------------------------- | ------------------------------------ |
| Start centralized training | `CentralizedTrainer(config).train()` |
| Start federated training   | `ServerApp(config).main()`           |
| Run single prediction      | `InferenceEngine().predict(image)`   |
| Generate heatmap           | `GradCAM(model)(image)`              |
| Query research agent       | `ArxivAgentEngine().stream(query)`   |
| Generate PDF report        | `generate_prediction_report(result)` |
