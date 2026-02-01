# Technical Architecture Report
## Hybrid Federated/Centralized Pneumonia Detection System with Extended Clinical Capabilities

**Final Year Project Technical Documentation**

---

## Executive Summary

This report documents the technical architecture of a Hybrid Federated/Centralized Pneumonia Detection System designed for chest X-ray analysis. The system was developed with a baseline scope focused on comparative analysis between centralized deep learning and privacy-preserving federated learning approaches. During development, two critical extensions were added to enhance clinical utility: (1) a production-grade Inference Engine for real-time model deployment, and (2) an Agentic AI System powered by Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to provide research-backed clinical explanations.

The architecture follows Clean Architecture principles with strict separation of concerns across API, Control, Domain, Boundary, and Entity layers. This modular design ensured that the extensions could be integrated without disrupting the core training infrastructure, demonstrating architectural adaptability for evolving project requirements.

---

## 6.3.4 Key EDA Findings Summary

### Finding 1: Class Imbalance
The 2.16:1 normal-to-pneumonia ratio requires stratified sampling and weighted loss functions to prevent the model from defaulting to the majority class.

### Finding 2: Data Completeness
100% metadata completeness and successful image file validation confirms data pipeline reliability with zero missing values post-cleaning.

### Finding 3: Non-Normal Distributions
Shapiro-Wilk tests reject normality for age and bounding box dimensions, justifying bootstrap confidence interval estimation and non-parametric statistical testing when comparing federated to centralized approaches.

## 6.4 Model Implementation

The federated pneumonia detection pipeline employs a pre-trained ResNet50 V2 backbone augmented with a custom classification head optimized for binary pneumonia detection. This architecture balances computational efficiency with radiographic pattern recognition, leveraging ImageNet pretraining while allowing domain-specific fine-tuning. Training operates within PyTorch Lightning, abstracting distributed concerns and enabling seamless switching between centralized and federated learning paradigms. Performance is rigorously evaluated across classification metrics (accuracy, precision, recall, F1) and ranking metrics (AUROC), with statistical robustness established through multi-run evaluation with different random seeds.

### 6.4.1 Algorithms/Models Used

**Architecture Design**

The primary model is ResNet50 V2 (ImageNet1K V2 pretrained) with a custom classification head. The backbone extracts hierarchical features from 256x256 RGB chest X-rays through residual connections and spatial pooling. Global average pooling condenses the feature maps (batch, 2048, 8, 8) to a flat vector (batch, 2048). The classification head implements three fully connected layers with ReLU activation: 2048 → 256 → 64 → 1, with configurable dropout (default 0.3) between layers to prevent overfitting. The output is a single logit per sample, passed to sigmoid for probability prediction or to BCEWithLogitsLoss for stable training.

**Model Statistics and Configuration**

The complete architecture contains approximately 23.5 million parameters. By default, the ResNet50 V2 backbone is frozen, preserving ImageNet pretraining and reducing computational cost. The custom head remains fully trainable, adapting to pneumonia-specific features. Optional fine-tuning unfreezes the last N layers of the backbone (configurable via fine_tune_layers_count), allowing supervised adaptation when domain divergence warrants it. Input tensors conform to (batch, 3, 256, 256) specification, and outputs are single-channel logits (batch, 1).

### 6.4.2 Training and Testing Procedures

**Data Splitting Strategy**

| Parameter | Configuration |
|-----------|---------------|
| Split Ratio | 80% Training / 20% Validation |
| Stratification | Enabled (preserves class distribution) |
| Random Seed | 42 |
| Pneumonia Prevalence | 31.61% (maintained in both sets) |
| Purpose | Prevent class balance mismatch between training and evaluation |

**Training Workflow**

| Parameter | Configuration |
|-----------|---------------|
| Step 1 | ZIP extraction or directory discovery |
| Step 2 | Metadata loading and validation |
| Step 3 | XRayDataModule instantiation (batching + augmentation) |
| Step 4 | ResNet50 V2 instantiation with custom head |
| Step 5 | Callback and logger setup (Weights and Biases) |
| Step 6 | Trainer configuration (GPU + mixed precision) |
| Step 7 | trainer.fit() execution |

**Optimization Configuration**

| Parameter | Configuration |
|-----------|---------------|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 (L2 regularization) |
| Batch Size | 32 |
| Loss Function | BCEWithLogitsLoss (sigmoid + BCE, numerically stable) |
| Class Weighting | Optional (pos_weight = count_negative / count_positive) |
| Class Imbalance Ratio | 2.16:1 (Normal:Pneumonia) |

**Scheduling and Early Stopping**

| Parameter | Configuration |
|-----------|---------------|
| Scheduler | ReduceLROnPlateau |
| Monitor Metric | Validation Recall |
| LR Reduction Factor | 0.5 (halves learning rate) |
| Scheduler Patience | 3 epochs |
| Early Stopping Patience | 5 epochs |
| Early Stopping Delta | 0.001 |
| Minimum Learning Rate | 1.0e-07 |

### 6.4.3 Hyperparameter Tuning

**Tuning Approach**

| Parameter | Default | Range / Description |
|-----------|---------|---------------------|
| Learning Rate | 0.001 | [1e-5, 1e-1] - Controls optimizer step size |
| Weight Decay | 0.0001 | [0, 1e-2] - L2 regularization strength |
| Dropout Rate | 0.3 | [0, 0.7] - Stochastic regularization in head |
| Batch Size | 32 | [8, 256] - Gradient noise vs. efficiency |
| Epochs | 10 | [1, 100] - Training budget (early stopping may terminate) |
| Fine-tune Layers | 5 | [0, ~100] - Backbone plasticity control |

**Configuration Management and Reproducibility**

| Parameter | Configuration |
|-----------|---------------|
| Config Class | ConfigManager |
| Config File | default_config.yaml |
| Seed Management | seed_manager.py |
| Multi-run Metrics | Mean, Standard Deviation, Min/Max |
| Purpose | Runtime modification + statistical significance testing |

### 6.4.4 Performance Metrics Used

**Classification Metrics**

| Metric | Formula | Clinical Significance |
|--------|---------|----------------------|
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | Measures overall correctness across all predictions |
| Precision | TP / (TP+FP) | Quantifies positive prediction reliability; critical when false positives trigger unnecessary clinical workup |
| Recall (Sensitivity) | TP / (TP+FN) | Captures pneumonia detection rate; directly impacts patient outcomes |
| F1 Score | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall; suitable when class imbalance exists |
| AUROC | Area under ROC curve | Measures rank-order discriminability; independent of classification threshold |

## 6.5 Version Control and Collaboration

Version control practices ensure reproducibility, enable collaboration, and maintain development history. The project employs Git for source control with GitHub as the remote repository.

### 6.5.1 Use of Git/GitHub

**Repository Structure**

The repository follows clean architecture with six top-level directories: federated_pneumonia_detection/ (main app), xray-vision-ai-forge/ (frontend), analysis/, documentation/, tests/, and Training_Sample_5pct/. The application package contains four layers: API (src/api/ - FastAPI endpoints), Boundary (src/boundary/ - SQLAlchemy models/CRUD), Control (src/control/ - centralized and federated training logic), and Entities (src/entities/ - domain models like ResNet architecture). Configuration resides in config/. The .gitignore excludes bytecode, virtual environments, datasets, checkpoints, and IDE metadata.

**Git Workflow**

Commit messages use conventional prefixes: feat, fix, refactor, docs, test. Branch naming uses dates for experiments (2025-11-01-experimental-feature) or feature identifiers (feature/comparative-analysis). The main branch holds production-ready code; feature branches merge only after testing and review.

## 6.6 Implementation Challenges and Solutions

| Challenge | Problem | Solution |
|-----------|---------|----------|
| 1. Class Imbalance | Models biased toward majority class; high accuracy but poor disease detection | Weighted binary cross-entropy loss with positive class weight = negative/positive sample ratio |
| 2. Memory Constraints | Out-of-memory errors when loading high-resolution datasets exceeding system RAM | Lazy loading via __getitem__; stores file paths only, loads images on-demand during batch construction |
| 3. Training Instability | Loss fluctuation and oscillating validation metrics preventing convergence | Early stopping (patience=5), ReduceLROnPlateau (patience=3, factor=0.5), AdamW with weight decay (0.0001) |
| 4. Format Variation | Pipeline failures from heterogeneous image formats, color modes, and quality levels | Defensive preprocessing with try-except fallbacks; PIL-based detection; automatic RGB conversion |
| 5. Configuration Complexity | Hyperparameters scattered across files; reproducibility and tracking difficulties | Centralized ConfigManager with hierarchical YAML; dot-notation access; version-controlled configs |
| 6. Real-time Monitoring | HTTP polling caused server load and latency in training progress visibility | WebSocket server-push architecture; background thread prevents blocking; JSON message protocol |
| 7. Reproducibility | Random initialization and shuffling caused variance across identical experiments | SeedManager seeds Python, NumPy, PyTorch CPU/CUDA; seed values persisted in config files |

## 6.7 Implementation Summary

| Component | Implementation | Rationale |
|-----------|---------------|-----------|
| Deep Learning | PyTorch + PyTorch Lightning | Dynamic graphs; medical imaging adoption; training abstraction |
| Federated Learning | Flower Framework | Minimal infrastructure; simulation and production modes |
| Backend API | FastAPI + Pydantic | Native async; auto OpenAPI docs; type validation |
| Frontend | React + TypeScript + Vite | Type safety; fast builds; reactive updates |

The implementation successfully addresses the core requirements of medical imaging classification while supporting both centralized and federated deployment scenarios. The ResNet50 V2 architecture provides a strong foundation for radiographic pattern recognition, while the modular training infrastructure enables comparative analysis between learning paradigms. The comprehensive metrics suite ensures clinical relevance, and the robust version control practices maintain experimental reproducibility across multiple runs.

---

Having established the technical foundation through careful data analysis, model implementation, and systematic resolution of implementation challenges, the following sections provide a comprehensive architectural overview of the Hybrid Federated/Centralized Pneumonia Detection System. The design decisions documented in Sections 6.4 through 6.7—specifically the ResNet50 V2 architecture with custom classification head, PyTorch Lightning training abstraction, and Flower-based federated learning integration—are formalized here into a layered system architecture. This architectural blueprint illustrates how the individual components coalesce into a production-ready system that balances computational efficiency with clinical utility, ensuring seamless transitions between centralized experimentation and privacy-preserving federated deployment while maintaining the rigorous performance standards required for medical imaging applications.

## 6.8.1 System Architecture Overview

### 6.8.1.1 High-Level Component Architecture

The system is organized into seven distinct layers, each with well-defined responsibilities:

**Layer 1: Client Device (Web Browser)**
React-based Dashboard UI for experiment management, real-time WebSocket client for training metrics streaming, and chat interface for AI-powered research assistance.

**Layer 2: API Layer (FastAPI)**
Training API orchestrates centralized and federated training, Inference API handles single and batch prediction requests, Chat API manages conversation sessions and streaming queries, and WebSocket Manager provides real-time bidirectional communication.

**Layer 3: Training Control Layer**
Training Coordinator (CentralizedTrainer) provides end-to-end centralized training orchestration, FL Server (ServerApp) coordinates Flower-based federated learning, FL Client (ClientApp) handles distributed client training nodes, and Custom Strategy (ConfigurableFedAvg) implements weighted aggregation with WebSocket broadcasting.

**Layer 4: Deep Learning Layer (Shared Components)**
LitResNetEnhanced provides PyTorch Lightning ResNet50 wrapper with progressive unfreezing, XRay DataModule handles data loading and augmentation pipeline, and Training Utilities include callbacks for checkpointing, early stopping, and real-time metrics.

**Layer 5: Inference Layer (ADDED)**
InferenceService provides unified facade for pneumonia detection operations, and Grad-CAM Visualizer generates attention heatmaps for explainability.

**Layer 6: Agentic AI Layer (ADDED)**
ChatAgent (ArxivAugmentedEngine) provides LLM-powered research assistant, RAG Pipeline (QueryEngine) implements hybrid BM25 and semantic retrieval, ArXiv Tool (MCPManager) integrates academic paper search, and Session Manager handles conversation history and context management.

**Layer 7: Data Persistence Layer**
Database Manager (RunCRUD) manages training run lifecycle, Metric CRUD (RunMetricCRUD) handles metrics storage with federated context, and File Manager manages checkpoint and results file operations.

### 6.8.1.2 Class Architecture

The class diagram reveals the inheritance hierarchy and dependency relationships across the system layers.

**API Layer Classes:**
ExperimentsRouter, InferenceRouter, and ChatRouter define FastAPI endpoints, while WebSocketManager serves as a singleton connection manager for real-time communication.

**Training Control Classes:**
CentralizedTrainer orchestrates complete training workflows, ServerApp and ClientApp implement Flower decorator-based federated learning applications, and ConfigurableFedAvg extends the base FedAvg strategy with custom aggregation logic.

**Domain Layer Classes:**
LitResNetEnhanced implements LightningModule with ResNet50 backbone, XRayDataModule provides LightningDataModule for dataset orchestration, and CustomImageDataset implements PyTorch Dataset with validation and error handling.

**Agentic AI Classes:**
ArxivAugmentedEngine provides research agent with RAG and ArXiv capabilities, QueryEngine implements hybrid retrieval with ensemble ranking, and MCPManager manages ArXiv MCP server lifecycle.

**Persistence Classes:**
RunCRUD and RunMetricCRUD implement SQLAlchemy CRUD operations for database entities including Run, RunMetric, ChatSession, and Message.

---

## 6.8.2 Baseline System: Core Training Infrastructure

### 6.8.2.1 Original Scope

The initial project scope focused exclusively on comparative analysis between centralized and federated learning approaches for pneumonia detection. The baseline system provided centralized training mode for single-machine training with full dataset access, federated training mode for distributed training simulating multi-hospital scenarios, comparative analytics for metrics collection and performance benchmarking, and WebSocket streaming for real-time training progress visualization.

### 6.8.2.2 Centralized Training Architecture

The CentralizedTrainer serves as the primary orchestrator for the centralized training workflow. The train() method implements the Template Method pattern, defining the skeleton of the training algorithm while delegating specific steps to helper functions.

```python
def train(self, source_path: str, experiment_name: str, ...):
    run_id = create_training_run(source_path, experiment_name, self.logger)
    image_dir, csv_path = self.data_source_extractor.extract_and_validate(...)
    train_df, val_df = prepare_dataset(csv_path, image_dir, self.config, ...)
    data_module = create_data_module(train_df, val_df, image_dir, self.config, ...)
    model, callbacks, metrics_collector = build_model_and_callbacks(...)
    trainer = build_trainer(self.config, callbacks, self.logs_dir, ...)
    trainer.fit(model, data_module)
    results = collect_training_results(trainer, model, metrics_collector, ...)
    return results
```

Key design decisions include PyTorch Lightning for abstracting training loop complexity and enabling callback integration, YAML-based configuration for experiment reproducibility, and modular callbacks for real-time metrics streaming without blocking training operations.

### 6.8.2.3 Federated Training Architecture

The federated system uses the Flower framework with custom extensions for coordination and aggregation. The ServerApp class coordinates the federation using decorator patterns to separate framework concerns from business logic.

```python
@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds: int = context.run_config["num-server-rounds"]
    num_clients: int = len(list(grid.get_node_ids()))
    run_id, _ = _initialize_database_run()
    global_model, arrays = _build_global_model(config_manager)
    strategy = _initialize_strategy(train_config, eval_config, run_id, num_rounds)
    result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=num_rounds, ...)
```

The ClientApp handles local training by loading data partitions and applying global model weights before local training execution.

```python
@app.train()
def train(msg: Message, context: Context):
    centerlized_trainer, config = _load_trainer_and_config()
    partition_id = context.node_id % configs["num_partitions"]
    partion_df = partioner.load_partition(partition_id)
    train_df, val_df = _prepare_partition_and_split(...)
    data_module = XRayDataModule(train_df=train_df, val_df=val_df, ...)
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state_dict)
    trainer.fit(model, data_module)
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(metrics_history)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
```

The ConfigurableFedAvg class extends the base FedAvg algorithm to incorporate custom aggregation logic, WebSocket broadcasting for real-time updates, and database persistence for metrics tracking.

```python
def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
    aggregated_metrics = super().aggregate_evaluate(server_round, replies_list)
    if aggregated_metrics:
        round_metrics = self._extract_round_metrics(aggregated_metrics)
        self.ws_sender.send_metrics({
            "round": server_round,
            "metrics": round_metrics,
        }, "round_metrics")
        self._persist_aggregated_metrics(server_round, round_metrics)
```

Key design decisions include the Strategy Pattern for customizable aggregation logic without framework modification, weighted aggregation using num_examples from clients for proportional weighting, and real-time broadcasting via WebSocket integration for live progress monitoring.

---

## 6.8.3 Extended Capabilities: Clinical Deployment & Research Assistance

### 6.8.3.1 Motivation for Extensions

During the development lifecycle, two critical gaps were identified. The Model Utility Gap indicated that the baseline system produced trained models but lacked production deployment capability, as a model's value is realized only through inference on new patient data. The Domain Knowledge Gap revealed that clinicians needed access to research literature and model explanations, but the baseline provided only raw predictions without contextual support. These extensions were added while maintaining architectural consistency, demonstrating the system's adaptability to evolving requirements.

### 6.8.3.2 Inference Engine (Production Deployment)

The Inference Engine enables real-time pneumonia detection on new chest X-ray images with explainability features. The InferenceService implements the Facade pattern by composing multiple specialized components behind a unified interface.

```python
class InferenceService:
    def __init__(self, engine: Optional[InferenceEngine] = None):
        self._engine = engine
        self.validator = ImageValidator()
        self.processor = ImageProcessor()
        self.batch_stats = BatchStatistics()
        self.logger = ObservabilityLogger()
    
    async def predict_batch(self, files: list) -> BatchInferenceResponse:
        self.check_ready_or_raise()
        max_batch_size = 500
        if len(files) > max_batch_size:
            raise HTTPException(status_code=400, detail="Maximum 500 images allowed")
        results = []
        for file in files:
            result = await self.process_single(file=file)
            results.append(result)
        summary = self.batch_stats.calculate(results=results, total_images=len(files))
        return BatchInferenceResponse(predictions=results, summary=summary)
```

Integration points include reusing LitResNetEnhanced from the training layer for code reuse, lazy-loaded singleton pattern for model efficiency, batch processing with configurable limits up to 500 images, and Grad-CAM integration for attention visualization and clinical trust.

### 6.8.3.3 Agentic AI System (Research Assistance)

The Agentic AI System provides LLM-powered research assistance with access to medical literature and system documentation. The ChatAgent combines multiple capabilities through the ArxivAugmentedEngine class.

```python
class ArxivAugmentedEngine(BaseAgent):
    def __init__(self, ...):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=1.0,
            max_tokens=2048,
        )
        try:
            self._query_engine = QueryEngine()
            self._rag_tool = create_rag_tool(self._query_engine)
        except Exception as e:
            logger.warning(f"RAG tool unavailable: {e}")
        self._history_manager = ChatHistoryManager()
```

The RAG Pipeline implements hybrid retrieval combining semantic and lexical search methods.

```python
class QueryEngine:
    def __init__(self, ...):
        self.vector_store_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10}
        )
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 10
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_store_retriever],
            weights=[0.5, 0.5],
        )
```

ArXiv integration is achieved through the MCP protocol via the MCPManager singleton.

```python
class MCPManager:
    _instance: Optional[MCPManager] = None
    _lock: asyncio.Lock = asyncio.Lock()
    
    async def initialize(self):
        server_config = {
            "arxiv": {
                "command": "arxiv-mcp-server",
                "args": [],
                "transport": "stdio",
            }
        }
        self._client = MultiServerMCPClient(server_config)
        self._tools = await self._client.get_tools()
        self._is_available = True
```

Integration points include maintaining a separate Agentic AI Layer for isolation from DL models, RAG retrieval from PostgreSQL vector store using PGVector, external research access via ArXiv tool, persistent conversation history through session management, and streaming responses via Server-Sent Events for responsive user experience.

---

## 6.8.4 Technical Implementation Details

### 6.8.4.1 Database Schema Design

The PostgreSQL schema supports both centralized and federated training contexts with flexible metric storage. The schema implements unified metric storage through the run_metrics table, which accommodates both centralized metrics without client or round context and federated metrics with client_id and round_id fields. This design avoids schema fragmentation while supporting both training modes. The context column enables semantic categorization of metrics distinguishing between global model versus local client metrics and aggregated versus raw values. Chat sessions are fully persisted enabling conversation resumption and audit trails for clinical decision support.

The core tables include the Run table with fields for id, training_mode, status, start_time, end_time, and source_path. The RunMetric table extends this with metric_name, metric_value, step, dataset_type, and nullable client_id and round_id fields for federated context. The Client table tracks federated participants with client_identifier and client_config fields. The ChatSession and Message tables enable persistent conversation storage with UUID-based session identification and role-based message categorization.

### 6.8.4.2 Architectural Patterns and Implementation

#### 6.8.4.2.1 Patterns Employed

The system implements eight fundamental architectural patterns. Clean Architecture ensures strict separation across API, Control, Domain, Boundary, and Entity layers keeping business logic independent of frameworks. The Strategy Pattern enables ConfigurableFedAvg to extend FedAvg without modifying the Flower framework. Singleton Pattern ensures single points of control for WebSocketManager, SessionManager, and MCPManager. Facade Pattern provides simplified interfaces through InferenceService shielding complexity from API endpoints. Repository Pattern abstracts database operations via CRUD classes enabling test mocking. Delegation Pattern promotes single responsibility by having LitResNetEnhanced delegate to StepLogic, MetricsHandler, and LossFactory. Factory Pattern creates training components through OptimizerFactory and LossFactory based on configuration. Decorator Pattern separates framework boilerplate from business logic in Flower applications.

#### 6.8.4.2.2 Key Design Decisions

Configuration-driven architecture externalizes all training parameters to YAML files enabling reproducible experiments, easy hyperparameter tuning, and separation of code from configuration. Shared components between modes ensure both centralized and federated training use identical LitResNetEnhanced and XRayDataModule classes with federated clients simply reusing centralized training logic on local partitions. This ensures consistent model behavior, code reuse, and fair comparison between approaches. Real-time metrics streaming via WebSocket provides sub-second latency enabling immediate anomaly detection, user engagement during training, and early stopping decisions. Lazy loading for the Inference Engine avoids loading models until the first request enabling fast server startup, memory efficiency, and graceful degradation. Hybrid retrieval for RAG combines BM25 lexical and PGVector semantic search with ensemble ranking ensuring keyword matches are captured while semantic similarity provides conceptual relevance.

#### 6.8.4.2.3 Pattern Implementation in Control Layer

The Control Layer implements core business logic through carefully orchestrated class interactions. The CentralizedTrainer class demonstrates the Template Method pattern defining the skeleton of the training algorithm while delegating specific steps to helper functions enabling consistent workflows with flexibility in data preparation, model configuration, and result collection.

```python
def train(self, source_path: str, experiment_name: str, ...):
    run_id = create_training_run(source_path, experiment_name, self.logger)
    image_dir, csv_path = self.data_source_extractor.extract_and_validate(...)
    train_df, val_df = prepare_dataset(csv_path, image_dir, self.config, ...)
    data_module = create_data_module(train_df, val_df, image_dir, self.config, ...)
    model, callbacks, metrics_collector = build_model_and_callbacks(...)
    trainer = build_trainer(self.config, callbacks, self.logs_dir, ...)
    trainer.fit(model, data_module)
    results = collect_training_results(trainer, model, metrics_collector, ...)
    return results
```

The federated learning architecture extends this through Flower's decorator-based approach. ServerApp coordinates federated rounds using the Decorator pattern where lifecycle hooks separate framework concerns from business logic enabling clean integration with the Flower protocol while maintaining architectural boundaries.

```python
@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds: int = context.run_config["num-server-rounds"]
    num_clients: int = len(list(grid.get_node_ids()))
    run_id, _ = _initialize_database_run()
    global_model, arrays = _build_global_model(config_manager)
    strategy = _initialize_strategy(train_config, eval_config, run_id, num_rounds)
    result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=num_rounds, ...)
```

ClientApp implements the Command pattern encapsulating local training operations as discrete messages transmitted across the network. Each client receives global model state, performs local training using CentralizedTrainer logic, and returns model updates with metrics ensuring consistency between modes while enabling distributed execution.

```python
@app.train()
def train(msg: Message, context: Context):
    centerlized_trainer, config = _load_trainer_and_config()
    partition_id = context.node_id % configs["num_partitions"]
    partion_df = partioner.load_partition(partition_id)
    train_df, val_df = _prepare_partition_and_split(...)
    data_module = XRayDataModule(train_df=train_df, val_df=val_df, ...)
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state_dict)
    trainer.fit(model, data_module)
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(metrics_history)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
```

ConfigurableFedAvg demonstrates the Strategy pattern extending base FedAvg to incorporate custom aggregation logic, WebSocket broadcasting, and database persistence allowing customization without modifying the Flower framework.

```python
def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
    aggregated_metrics = super().aggregate_evaluate(server_round, replies_list)
    if aggregated_metrics:
        round_metrics = self._extract_round_metrics(aggregated_metrics)
        self.ws_sender.send_metrics({
            "round": server_round,
            "metrics": round_metrics,
        }, "round_metrics")
        self._persist_aggregated_metrics(server_round, round_metrics)
```

#### 6.8.4.2.4 Pattern Implementation in Extended Capabilities

The Inference Engine demonstrates the Facade pattern providing a unified interface shielding the API layer from complexity of image validation, preprocessing, model inference, and results aggregation. InferenceService composes ImageValidator, ImageProcessor, BatchStatistics, and InferenceEngine behind a single clean interface handling both single-image and batch prediction scenarios.

```python
def __init__(self, engine: Optional[InferenceEngine] = None):
    self._engine = engine
    self.validator = ImageValidator()
    self.processor = ImageProcessor()
    self.batch_stats = BatchStatistics()
    self.logger = ObservabilityLogger()

async def predict_batch(self, files: list) -> BatchInferenceResponse:
    self.check_ready_or_raise()
    results = [await self.process_single(file=f) for f in files]
    summary = self.batch_stats.calculate(results=results, total_images=len(files))
```

The Agentic AI system employs the Strategy pattern through ArxivAugmentedEngine inheriting from BaseAgent to provide interchangeable agent behavior enabling support for different agent implementations without modifying chat API or session management components.

```python
def __init__(self, max_history: int = 10, query_engine: Optional[QueryEngine] = None):
    self._history_manager = ChatHistoryManager(max_history=max_history)
    self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)
    self._query_engine = query_engine or QueryEngine()
    self._rag_tool = create_rag_tool(self._query_engine)

async def query_stream(self, query: str, session_id: str, arxiv_enabled: bool = False):
    async for event in stream_query(self.llm, self._history_manager, 
                                    self._rag_tool, query, session_id):
        yield event
```

The WebSocket communication layer implements the Observer pattern with asynchronous broadcasting maintaining a registry of connected clients and broadcasting messages when training metrics become available decoupling the training control layer from transport concerns.

```python
async def handler(websocket: websockets.WebSocketServerProtocol) -> None:
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            await _broadcast_to_clients(message, websocket, connected_clients)
    finally:
        connected_clients.discard(websocket)
```

#### 6.8.4.2.5 Pattern Implementation in Persistence Layer

The Repository pattern is implemented through BaseCRUD providing a generic foundation for all database operations using Python's Generic type parameterization enabling type-safe specialization while maintaining consistent patterns across entity repositories.

```python
def create(self, db: Session, **kwargs) -> ModelType:
    db_obj = self.model(**kwargs)
    db.add(db_obj)
    db.flush()
    db.refresh(db_obj)
    return db_obj

def bulk_create(self, db: Session, objects: List[Dict[str, Any]]) -> List[ModelType]:
    db_objs = [self.model(**obj) for obj in objects]
    db.add_all(db_objs)
    db.flush()
    return db_objs
```

RunCRUD extends this foundation implementing a specialized repository for training run persistence encapsulating domain-specific logic including status transitions, metric persistence with federated context support, and eager loading of relationships to prevent N+1 query problems.

```python
def complete_run(self, db: Session, run_id: int, status: str = "completed"):
    run = db.query(self.model).filter(self.model.id == run_id).first()
    if not run:
        return None
    return self.update(db, run_id, end_time=datetime.now(), status=status)

def persist_metrics(self, db: Session, run_id: int, epoch_metrics: List[Dict]):
    client_id, round_id = self._resolve_federated_context(db, federated_context)
    metrics_to_persist = []
    for epoch_data in epoch_metrics:
        metrics_to_persist.extend(
            self._transform_epoch_to_metrics(epoch_data, run_id, client_id, round_id)
        )
    if metrics_to_persist:
        run_metric_crud.bulk_create(db, metrics_to_persist)
        db.commit()
```

Entity models implement the Active Record pattern through SQLAlchemy's declarative base defining data structures that repositories manage abstracting table schemas behind Python classes with type annotations for compile-time safety.

```python
class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    training_mode = Column(String(50), nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)
    metrics = relationship("RunMetric", back_populates="run")

class RunMetric(Base):
    __tablename__ = "run_metrics"
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
```

---

## 6.8.5 Conclusion

The Hybrid Federated/Centralized Pneumonia Detection System demonstrates a mature, production-ready architecture suitable for both research and clinical deployment. The baseline system successfully implements comparative training infrastructure, while the extensions (Inference Engine and Agentic AI) significantly enhance clinical utility without architectural compromise.

Key achievements include architectural consistency ensuring extensions follow established patterns and maintain codebase coherence, separation of concerns through Clean Architecture enabling independent evolution of components, and clinical utility transformation from a training tool into a deployment-ready system with inference and research assistance features. The system successfully bridges the gap between research experimentation and clinical application, providing a foundation for future extensions such as multi-modal data integration, automated hyperparameter tuning, or integration with hospital PACS systems.

---

## Appendix A: Technology Stack

**Deep Learning:** PyTorch 2.x, PyTorch Lightning, TorchVision
**Federated Learning:** Flower (FL) Framework
**Web Framework:** FastAPI, Uvicorn
**Database:** PostgreSQL 15+, SQLAlchemy 2.x, PGVector
**Agentic AI:** LangChain, Google Gemini, HuggingFace Embeddings
**Data Processing:** Pandas, NumPy, Pillow, Albumentations
**Frontend:** React 18, TypeScript, WebSocket API
**Infrastructure:** Docker, Docker Compose (optional)

## Appendix B: File Structure Reference

```
federated_pneumonia_detection/
├── src/
│   ├── api/endpoints/          # FastAPI routers
│   ├── control/
│   │   ├── dl_model/           # Centralized training
│   │   ├── federated_new_version/  # FL server/client
│   │   ├── model_inferance/    # Inference engine (ADDED)
│   │   └── agentic_systems/    # AI assistant (ADDED)
│   ├── entities/               # Domain models (datasets)
│   ├── boundary/
│   │   ├── CRUD/              # Database operations
│   │   └── models/            # SQLAlchemy entities
│   └── internals/              # Utilities, transforms, logging
├── docs/architecture/          # Diagrams and documentation
└── config/                     # YAML configurations
```

---

*End of Technical Architecture Report*
