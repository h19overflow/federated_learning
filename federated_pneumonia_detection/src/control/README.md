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
            Utils["utils/<br/>Supporting Services"]
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

    subgraph Utils["utils/<br/>Utilities"]
        Config["config_loader.py"]
        Data["data_processing.py"]
        Transforms["image_transforms.py"]
    end

    CT -->|Use| Models
    CT -->|Use| Shared
    Core -->|Use| Models
    Core -->|Use| Shared
    Part -->|Use| Models

    CT -->|Import| Utils
    Core -->|Import| Utils
    MAS -->|Import| Utils

    style Control fill:#f3e5f5
    style DL fill:#f8bbd0
    style Fed fill:#c8e6c9
    style Agent fill:#fff9c4
    style Shared fill:#e8f5e9
    style Utils fill:#f0f4c3
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

### Flow 1: Centralized Training Execution

```mermaid
graph TD
    A["API: POST /experiments/centralized<br/>+ dataset.zip"] -->|Queue| B["Background Task"]
    B -->|Initialize| C["CentralizedTrainer"]

    C -->|Load| D["Configuration<br/>default_config.yaml"]
    C -->|Extract| E["DataSourceHandler<br/>ZIP → Images"]
    C -->|Create| F["XRayDataModule<br/>Train/Val/Test"]
    C -->|Create| G["LitResNet<br/>PyTorch Lightning"]

    G -->|Callbacks| H["CustomCallbacks<br/>Metrics, EarlyStopping"]

    H -->|Every Epoch| I["Metrics Collection<br/>Loss, Accuracy, F1..."]
    I -->|Send| J["WebSocketSender<br/>epoch_end message"]
    J -->|Relay| K["WebSocket Server<br/>ws://localhost:8765"]
    K -->|Broadcast| L["React Frontend<br/>Charts Update"]

    I -->|Store| M["MetricsFilePersister<br/>JSON/CSV files"]

    G -->|Training| N["Model Optimization<br/>Forward/Backward Pass"]
    N -->|Best Weights| O["Save Checkpoint<br/>PyTorch .pt"]

    G -->|Final| P["Evaluate on Test Set<br/>Confusion Matrix"]
    P -->|Results| Q["Database Persist<br/>PostgreSQL"]

    C -->|Signal| R["send_training_end<br/>+ run_id"]
    R -->|Relay| K

    L -->|Polling| S["API: GET /api/runs/{run_id}/metrics"]
    S -->|Query| Q
    L -->|Display| T["Final Results Page<br/>Plots & Metrics"]

    style A fill:#ffe0b2
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style K fill:#f8bbd0
    style L fill:#e1f5ff
    style Q fill:#e8f5e9
    style T fill:#c8e6c9
```

### Flow 2: Federated Learning Execution

```mermaid
graph TD
    A["API: POST /experiments/federated<br/>+ dataset.zip, num_rounds=15"] -->|Queue| B["Background Task"]
    B -->|Extract| C["Data Preparation<br/>Load metadata"]
    C -->|Partition| D["DataPartitioner<br/>IID/Non-IID/Stratified"]

    D -->|Initialize| E["Flower ServerApp<br/>core/server_app.py"]
    E -->|Create| F["LitResNet<br/>Global Model"]
    E -->|Create| G["ConfigurableFedAvg<br/>Aggregation Strategy"]

    E -->|Signal| H["send_training_mode<br/>is_federated=True"]
    H -->|Relay| I["WebSocket Server"]
    I -->|Broadcast| J["React Frontend"]

    loop "For each Round (1 to 15)"
        E -->|Distribute| K["Broadcast Weights<br/>+ Config to Clients"]
        K -->|Receive| L["ClientApp ×N<br/>core/client_app.py"]

        par "Parallel Client Execution"
            L -->|Load| M["Data Partition<br/>Client 0"]
            M -->|Train| N["LitResNet<br/>Local Training"]
            N -->|Compute| O["Training Metrics<br/>Loss, Accuracy"]
        and
            L -->|Load| P["Data Partition<br/>Client 1"]
            P -->|Train| Q["LitResNet<br/>Local Training"]
            Q -->|Compute| R["Training Metrics"]
        and
            L -->|Load| S["Data Partition<br/>Client N"]
            S -->|Train| T["LitResNet<br/>Local Training"]
            T -->|Compute| U["Training Metrics"]
        end

        O -->|Return| V["Aggregate Weights<br/>FedAvg Strategy"]
        R -->|Return| V
        U -->|Return| V

        V -->|Aggregated| W["Server-Side Evaluation<br/>Test Set"]
        W -->|Compute| X["Metrics & CM<br/>Accuracy, Precision, Recall"]
        X -->|Persist| Y["Database<br/>server_evaluations table"]

        V -->|Signal| Z["send_round_metrics<br/>Aggregated metrics"]
        Z -->|Relay| I
        I -->|Update| J
    end

    E -->|Mark Complete| Y
    E -->|Signal| AA["send_training_end<br/>+ run_id"]
    AA -->|Relay| I

    J -->|Query| AB["API: /api/runs/{id}/federated-rounds"]
    AB -->|Fetch| Y
    J -->|Display| AC["Federated Results<br/>Per-round charts"]

    style A fill:#ffe0b2
    style E fill:#c8e6c9
    style L fill:#bbdefb
    style I fill:#f8bbd0
    style J fill:#e1f5ff
    style Y fill:#e8f5e9
    style AC fill:#c8e6c9
```

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

    style A fill:#fff3e0
    style B fill:#fff3e0
    style C fill:#fff9c4
    style G fill:#fff9c4
    style J fill:#fff9c4
    style M fill:#fff9c4
    style Q fill:#f8bbd0
    style S fill:#e1f5ff
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

    subgraph Utils["Utilities"]
        DH["DataSourceHandler<br/>ZIP extraction"]
        MC["MetricsCollector<br/>Aggregation"]
        WS["WebSocketSender<br/>Real-time stream"]
        MFP["MetricsFilePersister<br/>Local storage"]
    end

    CT_Train -->|Creates| LR_Init
    CT_Train -->|Creates| XD_Setup
    CT_Setup -->|Uses| Utils

    LR_Loss -->|Call| MC
    LR_Val -->|Call| MC

    MC -->|Send| WS
    MC -->|Save| MFP

    CT_Train -->|Extract| DH

    style CentralizedTrainer fill:#f8bbd0
    style LitResNet fill:#f3e5f5
    style XRayDataModule fill:#f3e5f5
    style Utils fill:#fff9c4
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

    style ServerApp fill:#c8e6c9
    style ClientApp fill:#bbdefb
    style Strategy fill:#c8e6c9
    style DataPart fill:#fff9c4
    style Utils_FL fill:#fff9c4
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

    style A fill:#ffe0b2
    style B fill:#fff3e0
    style C fill:#f0f4c3
    style D fill:#f0f4c3
    style E fill:#f0f4c3
    style H fill:#f3e5f5
    style K fill:#fff9c4
    style L fill:#f8bbd0
    style N fill:#e1f5ff
    style P fill:#e8f5e9
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
| **Configuration** | [../../config/default_config.yaml](../../config/default_config.yaml) |

