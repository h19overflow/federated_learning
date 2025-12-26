# Control Layer - Training Orchestration & Business Logic

**Purpose**: Orchestrates all training operations, model management, and data processing. Houses centralized training, federated learning, and research assistance systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Module Architecture](#module-architecture)
3. [Functional Flows](#functional-flows)
4. [Component Hierarchy](#component-hierarchy)
5. [Module Details](#module-details)
6. [Data Flow](#data-flow)

---

## Overview

The Control Layer contains the core business logic for:
- **Centralized Training**: Single-machine PyTorch Lightning training
- **Federated Learning**: Multi-client distributed training with Flower framework
- **Agentic Systems**: Research assistance with Arxiv + RAG

**Key Principles**:
- **Separation of Concerns**: Training, data, utilities kept separate
- **Reusability**: Shared components (LitResNet, XRayDataModule) used across modes
- **Configuration-Driven**: All parameters externalized to YAML config
- **Async Integration**: WebSocket metrics streaming in real-time

---

## Module Architecture

### High-Level Structure

```mermaid
graph TB
    subgraph Control["🎮 Control Layer<br/>src/control/"]
        subgraph DL["dl_model<br/>Centralized Training"]
            CT["CentralizedTrainer<br/>Main Orchestrator"]
            Utils_DL["utils/<br/>Supporting Services"]
        end

        subgraph Fed["federated_new_version<br/>Federated Learning"]
            Core["core/<br/>Flower Components"]
            Part["partioner.py<br/>Data Partitioning"]
        end

        subgraph Agent["agentic_systems<br/>Research Assistant"]
            MAS["multi_agent_systems/<br/>LangChain Agents"]
            RAG["pipelines/rag/<br/>Document Processing"]
        end
    end

    subgraph Shared["entities/<br/>Shared Models"]
        Models["Domain Objects<br/>Entities"]
    end

    subgraph Utils_Shared["utils/<br/>Utilities"]
        Config["config_loader.py"]
        Data["data_processing.py"]
        Transforms["image_transforms.py"]
    end

    CT -->|Use| Models
    CT -->|Use| Shared
    Core -->|Use| Models
    Core -->|Use| Shared
    Part -->|Use| Models

    CT -->|Import| Utils_Shared
    Core -->|Import| Utils_Shared
    MAS -->|Import| Utils_Shared

    %% Styling
    classDef control fill:#2962FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef shared fill:#D50000,stroke:#fff,stroke-width:2px,color:#fff;
    classDef utils fill:#00C853,stroke:#fff,stroke-width:2px,color:#fff;
    
    class Control,DL,Fed,Agent control;
    class Shared,Models shared;
    class Utils_Shared,Config,Data,Transforms utils;
```

### Complete Module Tree

```
control/
├── dl_model/                          # Centralized Training
│   ├── centralized_trainer.py         # Main orchestrator
│   ├── README.md                      # Centralized docs
│   └── utils/
│       ├── data/
│       │   ├── data_source_handler.py # ZIP extraction
│       │   ├── metrics_file_persister.py # Metrics storage
│       │   ├── websocket_metrics_sender.py # Real-time relay
│       │   └── README.md              # WebSocket docs
│       ├── model/
│       │   ├── lit_resnet.py          # PyTorch Lightning module
│       │   ├── xray_data_module.py    # Data loading
│       │   ├── metrics_collector.py   # Metrics aggregation
│       │   ├── training_callbacks.py  # PL callbacks
│       │   └── custom_image_dataset.py # Image loading
│       └── callbacks/
│           └── custom_callbacks.py    # Training callbacks
│
├── federated_new_version/             # Federated Learning (Flower)
│   ├── core/
│   │   ├── server_app.py              # Flower ServerApp
│   │   ├── client_app.py              # Flower ClientApp
│   │   ├── custom_strategy.py         # FedAvg strategy
│   │   ├── server_evaluation.py       # Server-side eval
│   │   └── utils.py                   # FL utilities
│   ├── partioner.py                   # Data partitioner
│   ├── toml_adjustment.py             # Config sync
│   ├── pyproject.toml                 # Flower config
│   └── README.md                      # FL docs
│
├── agentic_systems/                   # Research Assistance
│   ├── multi_agent_systems/
│   │   └── chat/
│   │       ├── arxiv_agent.py         # Arxiv search agent
│   │       ├── arxiv_agent_prompts.py # Agent prompts
│   │       ├── mcp_manager.py         # MCP protocol handler
│   │       ├── retriver.py            # Query engine
│   │       └── tools/
│   │           └── rag_tool.py        # RAG tool wrapper
│   └── pipelines/
│       └── rag/
│           └── pipeline.py            # PDF processing pipeline
│
└── [Other utilities and configs]
```

---

## Functional Flows

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

### Federated Learning - Part 1: Initialization

```mermaid
sequenceDiagram
    autonumber
    participant API as FastAPI
    participant Server as FL Server
    participant DB as Database
    participant WS as WebSocket

    API->>Server: Start Federated Session
    activate Server
    
    Server->>DB: Create New Run
    Server->>WS: Broadcast "FL Mode Started"
    
    Server->>Server: Initialize Global Model
    Server->>Server: Configure Strategy (FedAvg)
    
    deactivate Server
```

### Federated Learning - Part 2: Round Execution

```mermaid
sequenceDiagram
    autonumber
    participant Server as FL Server
    participant Client as FL Client
    participant DB as Database
    participant WS as WebSocket

    Note over Server, Client: Start of Round N
    
    Server->>Client: Send Global Weights & Config
    activate Client
    
    Client->>Client: Train on Local Partition
    Client->>Client: Evaluate on Local Val Set
    
    Client-->>Server: Return Updated Weights & Metrics
    deactivate Client
    
    Note over Server: Aggregation & Eval
    Server->>Server: Aggregate Updates (FedAvg)
    Server->>Server: Evaluate on Server Test Set
    
    Server->>DB: Persist Round Metrics
    Server->>WS: Broadcast Round Metrics
```

---

## Component Hierarchy

### Centralized Training Components

```mermaid
graph TB
    subgraph CentralizedTrainer["CentralizedTrainer<br/>Main Orchestrator"]
        CT_Init["__init__<br/>- Load config<br/>- Setup directories"]
        CT_Train["train()<br/>- Coordinate full flow<br/>- Error handling"]
        CT_Setup["_setup_trainer<br/>- Create PyTorch trainer<br/>- Configure callbacks"]
    end

    subgraph LitResNet["LitResNet<br/>PyTorch Lightning Module"]
        LR_Init["__init__<br/>- Model architecture<br/>- Loss, optimizers"]
        LR_Forward["forward()<br/>- Forward pass<br/>- Predictions"]
        LR_Loss["training_step()<br/>- Compute loss<br/>- Metrics"]
        LR_Val["validation_step()<br/>- Validation pass<br/>- Val metrics"]
    end

    subgraph XRayDataModule["XRayDataModule<br/>Data Loading"]
        XD_Setup["setup()<br/>- Load metadata<br/>- Create splits"]
        XD_Train["train_dataloader()"]
        XD_Val["val_dataloader()"]
        XD_Test["test_dataloader()"]
    end

    subgraph Utils_Comp["Utilities"]
        DH["DataSourceHandler<br/>ZIP extraction"]
        MC["MetricsCollector<br/>Aggregation"]
        WS["WebSocketSender<br/>Real-time stream"]
        MFP["MetricsFilePersister<br/>Local storage"]
    end

    CT_Train -->|Creates| LR_Init
    CT_Train -->|Creates| XD_Setup
    CT_Setup -->|Uses| Utils_Comp

    LR_Loss -->|Call| MC
    LR_Val -->|Call| MC

    MC -->|Send| WS
    MC -->|Save| MFP

    CT_Train -->|Extract| DH

    %% Styling
    classDef main fill:#FF6F00,stroke:#fff,stroke-width:2px,color:#fff;
    classDef model fill:#2962FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef util fill:#00C853,stroke:#fff,stroke-width:2px,color:#fff;

    class CentralizedTrainer main;
    class LitResNet,XRayDataModule model;
    class Utils_Comp,DH,MC,WS,MFP util;
```

### Federated Learning Components

```mermaid
graph TB
    subgraph ServerApp["ServerApp<br/>Flower Server"]
        SA_Init["__init__<br/>- Load config<br/>- Create model"]
        SA_Main["main()<br/>- Execute FL rounds<br/>- Orchestrate clients"]
        SA_Eval["_server_evaluation()<br/>- Evaluate on test set<br/>- Confusion matrix"]
    end

    subgraph ClientApp["ClientApp<br/>Flower Client"]
        CA_Fit["fit()<br/>- Receive weights<br/>- Train locally<br/>- Return updates"]
        CA_Eval["evaluate()<br/>- Validate model<br/>- Return metrics"]
    end

    subgraph Strategy["ConfigurableFedAvg<br/>Custom Strategy"]
        ST_Config["configure_train()<br/>- Prepare client config"]
        ST_Agg["aggregate_fit()<br/>- FedAvg aggregation<br/>- Weighted by samples"]
        ST_Agg_E["aggregate_evaluate()<br/>- Aggregate metrics"]
    end

    subgraph DataPart["DataPartitioner<br/>partioner.py"]
        DP_IID["partition_data_iid()"]
        DP_NonIID["partition_data_non_iid()"]
        DP_Strat["partition_data_stratified()"]
    end

    subgraph Utils_FL["Utilities"]
        Utils_Prep["_prepare_partition()<br/>Load partition"]
        Utils_Metrics["_extract_metrics()<br/>Parse metrics"]
        Utils_Persist["_persist_evaluations()<br/>Database save"]
    end

    SA_Main -->|Creates| ClientApp
    SA_Main -->|Uses| Strategy
    SA_Main -->|Uses| DataPart
    SA_Main -->|Uses| Utils_FL

    CA_Fit -->|Receives| Utils_Prep
    Strategy -->|Aggregates| CA_Fit
    Strategy -->|Persists| Utils_Persist

    SA_Eval -->|Save| Utils_Persist

    %% Styling
    classDef server fill:#AA00FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef client fill:#007BFF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef strat fill:#FF6F00,stroke:#fff,stroke-width:2px,color:#fff;

    class ServerApp server;
    class ClientApp client;
    class Strategy,DataPart,Utils_FL strat;
```

---

## Module Details

### 1. Centralized Training (`dl_model/`)

**Purpose**: Single-machine training orchestration

**Key Files**:
- `centralized_trainer.py`: Main orchestrator
- `utils/data/`: Data handling and WebSocket
- `utils/model/`: Model and metrics

**Key Features**:
- PyTorch Lightning training loop
- Automatic checkpointing
- Real-time metrics streaming
- Early stopping support

**See**: [dl_model/README.md](dl_model/README.md) for detailed documentation

---

### 2. Federated Learning (`federated_new_version/`)

**Purpose**: Multi-client distributed training with Flower

**Key Files**:
- `core/server_app.py`: Server orchestration
- `core/client_app.py`: Client training logic
- `core/custom_strategy.py`: FedAvg aggregation
- `partioner.py`: Data partitioning

**Key Features**:
- Privacy-preserving training
- Multiple data distribution strategies (IID, Non-IID, Stratified)
- Server-side evaluation
- Real-time round metrics

**See**: [federated_new_version/README.md](federated_new_version/README.md) for detailed documentation

---

### 3. Agentic Systems (`agentic_systems/`)

**Purpose**: Research assistance with LangChain

**Key Files**:
- `multi_agent_systems/chat/arxiv_agent.py`: Arxiv search
- `multi_agent_systems/chat/mcp_manager.py`: MCP protocol
- `pipelines/rag/pipeline.py`: Local RAG

**Key Features**:
- Arxiv paper search via MCP
- Local document RAG
- Contextual responses based on training results
- Session-based conversation history

### Agentic Systems

Harnesses LLMs for research assistance and RAG-based document retrieval.

- **Arxiv Agent**: [src/control/agentic_systems/multi_agent_systems/chat/arxiv_agent.py](agentic_systems/multi_agent_systems/chat/arxiv_agent.py) - Searches medical literature.
- **RAG Pipeline**: [src/control/agentic_systems/pipelines/rag/pipeline.py](agentic_systems/pipelines/rag/pipeline.py) - Processes local PDFs and medical reports.
- **MCP Manager**: [src/control/agentic_systems/multi_agent_systems/chat/mcp_manager.py](agentic_systems/multi_agent_systems/chat/mcp_manager.py) - Handles the Model Context Protocol for tool use.

### Flow 3: Research Assistance (Agentic System)

```mermaid
graph TD
    A["User Query<br/>via Frontend"] -->|POST /chat/query| B["Chat Endpoint<br/>chat_endpoints.py"]
    B -->|Load| C["ArxivAgent<br/>LangChain Agent"]
    B -->|Load| D["RAG Pipeline<br/>Local Documents"]

    C -->|Prepare| E["Agent Prompt<br/>System Message"]
    E -->|With Context| F["User Query<br/>+ Run Metadata"]

    F -->|Execute| G["LangChain Agentic Loop"]
    G -->|Think| H["Reason about<br/>Best approach"]

    H -->|Decide| I{Use External<br/>Tools?}

    I -->|Yes - RAG| J["RAG Tool<br/>Local document search"]
    J -->|Query| K["Vector DB<br/>Embeddings"]
    K -->|Retrieve| L["Relevant Documents<br/>Context"]
    L -->|Return| G

    I -->|Yes - Arxiv| M["Arxiv Tool<br/>MCP Protocol"]
    M -->|Query| N["Arxiv API<br/>Paper Search"]
    N -->|Results| O["Paper Metadata<br/>Title, Abstract, URL"]
    O -->|Return| G

    I -->|No| P["Generate Response<br/>Using context"]

    H -->|Repeat| G
    P -->|Format| Q["ChatResponse<br/>Answer + Sources"]
    Q -->|Return| B
    B -->|HTTP| R["React Frontend"]
    R -->|Display| S["Chat Interface<br/>Answer + Links"]

    %% Styling
    classDef user fill:#6200EA,stroke:#fff,stroke-width:2px,color:#fff;
    classDef endpoint fill:#0091EA,stroke:#fff,stroke-width:2px,color:#fff;
    classDef logic fill:#AA00FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef tool fill:#FF6F00,stroke:#fff,stroke-width:2px,color:#fff;

    class A user;
    class B,R,S endpoint;
    class C,D,E,F,G,H,I,P,Q logic;
    class J,K,L,M,N,O tool;
```

---

## Data Flow

### Complete Training Data Flow

```mermaid
graph LR
    A["Input: dataset.zip<br/>+ config.yaml"] -->|Extract| B["Images<br/>+ Metadata CSV"]
    B -->|Load| C["Pandas DataFrame<br/>patient_id, target, filename"]
    C -->|Transform| D["XRayDataModule<br/>Train/Val/Test split"]

    D -->|Images| E["CustomImageDataset<br/>On-the-fly transform"]
    E -->|Transform| F["Augmentation<br/>Rotation, Flip, Normalize"]
    F -->|Batching| G["DataLoader<br/>Batch processing"]

    G -->|Batch| H["LitResNet<br/>Model forward pass"]
    H -->|Predictions| I["Loss & Metrics<br/>BCELoss + AUC"]
    I -->|Backprop| J["Gradient Computation<br/>Adam optimizer"]
    J -->|Update| H

    I -->|Collect| K["MetricsCollector<br/>Per-epoch aggregation"]
    K -->|Send| L["WebSocketSender<br/>JSON payload"]
    L -->|Stream| M["WebSocket Server<br/>ws://localhost:8765"]
    M -->|Broadcast| N["React Frontend<br/>Real-time update"]

    K -->|Store| O["MetricsFilePersister<br/>Local JSON/CSV"]
    K -->|Persist| P["PostgreSQL<br/>run_metrics table"]

    H -->|Best Model| Q["Checkpointing<br/>PyTorch .pt"]
    Q -->|Save| R["File Storage<br/>models/checkpoints/"]

    %% Styling
    classDef input fill:#6200EA,stroke:#fff,stroke-width:2px,color:#fff;
    classDef process fill:#0091EA,stroke:#fff,stroke-width:2px,color:#fff;
    classDef model fill:#2962FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef output fill:#00C853,stroke:#fff,stroke-width:2px,color:#fff;

    class A,B input;
    class C,D,E,F,G process;
    class H,I,J model;
    class K,L,M,N,O,P,Q,R output;
```

---

## Related Documentation

| Component | Documentation |
|-----------|---------------|
| **Centralized Training** | [dl_model/README.md](dl_model/README.md) |
| **Federated Learning** | [federated_new_version/README.md](federated_new_version/README.md) |
| **WebSocket Metrics** | [dl_model/utils/data/README.md](dl_model/utils/data/README.md) |
| **API Layer** | [../api/README.md](../api/README.md) |
| **System Architecture** | [../../README.md](../../README.md) |
| **Configuration** | [../../config/README.md](../../config/README.md) |

