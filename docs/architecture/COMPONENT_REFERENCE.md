# Architecture Component & Class Reference

## Overview

This document provides a concise technical reference for all major components and classes in the Federated Pneumonia Detection System. Each entry includes a single-sentence purpose description, key methods, a critical code snippet, and dependencies.

---

## Component Diagram Reference

### 1. Client Device Layer

**Dashboard UI (React)**
- **Purpose**: React-based frontend for experiment management and monitoring.
- **Key Features**: Training initiation, real-time metrics visualization, chat interface.
- **Dependencies**: WebSocket client, REST API clients.

**Real-time Monitor (WebSocket Client)**
- **Purpose**: Establishes WebSocket connection for live training metrics streaming.
- **Dependencies**: Browser WebSocket API, backend WebSocket server.

---

### 2. API Layer

**Training API (FastAPI)**
- **Purpose**: HTTP endpoints for centralized and federated training orchestration.
- **Key Endpoints**: `POST /experiments/centralized/train`, `POST /experiments/federated/train`.
- **Code Snippet**:
```python
@router.post("/train")
async def start_centralized_training(
    background_tasks: BackgroundTasks,
    data_zip: UploadFile = File(...),
) -> Dict[str, Any]:
    source_path = await prepare_zip(data_zip, logger, experiment_name)
    background_tasks.add_task(
        run_centralized_training_task,
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
    )
```

**Inference API (FastAPI)**
- **Purpose**: Single and batch pneumonia detection inference endpoints.
- **Key Endpoint**: `POST /api/inference/predict`.
- **Code Snippet**:
```python
@router.post("/predict", response_model=InferenceResponse)
async def predict(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    service.check_ready_or_raise()
    return await service.predict_single(file=file)
```

**Chat API (FastAPI)**
- **Purpose**: Session management and streaming query endpoints for AI assistant.
- **Key Endpoints**: `GET /chat/sessions`, `POST /chat/query/stream`.
- **Dependencies**: SessionManager, ChatAgent.

**WebSocket Manager**
- **Purpose**: Manages WebSocket connections for real-time metrics broadcasting.
- **Key Methods**: `start_websocket_server_thread()`, `handler()`, `_broadcast_to_clients()`.
- **Code Snippet**:
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

---

### 3. Training Control Layer

**Training Coordinator (CentralizedTrainer)**
- **Purpose**: Orchestrates end-to-end centralized training workflow.
- **Key Methods**: `train()`, `_load_config()`, `get_training_status()`.
- **Code Snippet**:
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

**FL Server (ServerApp)**
- **Purpose**: Flower server orchestrating federated learning rounds and client coordination.
- **Key Methods**: `main()`, `lifespan()`, `_initialize_database_run()`.
- **Code Snippet**:
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

**FL Client (ClientApp)**
- **Purpose**: Client-side training handler for local data partition training.
- **Key Methods**: `train()`, `evaluate()`.
- **Code Snippet**:
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

**Custom Strategy (ConfigurableFedAvg)**
- **Purpose**: Custom Flower strategy extending FedAvg with WebSocket broadcasting and DB persistence.
- **Key Methods**: `configure_train()`, `aggregate_train()`, `aggregate_evaluate()`.
- **Code Snippet**:
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

---

### 4. Deep Learning Layer (Shared)

**LitResNetEnhanced (PyTorch Lightning)**
- **Purpose**: Enhanced ResNet50-based pneumonia detection model with progressive unfreezing.
- **Key Methods**: `forward()`, `training_step()`, `configure_optimizers()`, `progressive_unfreeze()`.
- **Code Snippet**:
```python
def __init__(self, config=None, base_model_weights=None, ...):
    super().__init__()
    self.save_hyperparameters(ignore=["config", "base_model_weights", ...])
    self.model = ResNetWithCustomHead(
        config=self.config,
        base_model_weights=base_model_weights,
        num_classes=num_classes,
        dropout_rate=self.config.get("experiment.dropout_rate", 0.5),
        fine_tune_layers_count=self.config.get("experiment.fine_tune_layers_count", 0),
    )

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x)
```

**XRay DataModule**
- **Purpose**: Lightning DataModule for X-ray dataset orchestration.
- **Key Methods**: `setup()`, `train_dataloader()`, `val_dataloader()`.
- **Code Snippet**:
```python
def setup(self, stage: Optional[str] = None) -> None:
    seed = self.config.get("experiment.seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_transforms = create_training_transforms(self.transform_builder, ...)
    val_transforms = create_validation_transforms(self.transform_builder, ...)
    if stage in ("fit", None):
        self.train_dataset = create_dataset(...)
        self.val_dataset = create_dataset(...)
```

**Training Utilities**
- **Purpose**: Collection of callbacks for training monitoring and control.
- **Key Callbacks**: `HighestValRecallCallback`, `BatchMetricsCallback`, `GradientMonitorCallback`, `ProgressiveUnfreezeCallback`.
- **Code Snippet** (HighestValRecallCallback):
```python
class HighestValRecallCallback(pl.Callback):
    def __init__(self):
        self.best_recall = 0.0
    
    def on_validation_epoch_end(self, trainer, pl_module):
        current_recall = trainer.callback_metrics.get("val_recall", 0.0)
        if current_recall > self.best_recall:
            self.best_recall = current_recall
            self.logger.info(f"New best validation recall: {self.best_recall:.4f}")
```

---

### 5. Inference Layer

**InferenceService**
- **Purpose**: Unified facade orchestrating all pneumonia detection inference operations.
- **Key Methods**: `predict()`, `process_single()`, `predict_batch()`.
- **Code Snippet**:
```python
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
```

**Grad-CAM Visualizer**
- **Purpose**: Generates attention heatmaps for model predictions.
- **Dependencies**: PyTorch, OpenCV.

---

### 6. Agentic AI Layer

**ChatAgent (ArxivAugmentedEngine)**
- **Purpose**: Research agent combining RAG with ArXiv paper search for conversational AI.
- **Key Methods**: `query_stream()`, `_query_internal()`, `stream()`.
- **Code Snippet**:
```python
def __init__(self, ...):
    self.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1.0,
        max_tokens=2048,
    )
    try:
        if query_engine:
            self._query_engine = query_engine
        else:
            self._query_engine = QueryEngine()
        self._rag_tool = create_rag_tool(self._query_engine)
    except Exception as e:
        logger.warning(f"RAG tool unavailable: {e}")
```

**RAG Pipeline (QueryEngine)**
- **Purpose**: Hybrid retrieval combining BM25 lexical and PGVector semantic search.
- **Key Methods**: `query()`, `query_with_history()`, `query_with_history_stream()`.
- **Code Snippet**:
```python
def __init__(self, ...):
    self.vector_store_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
    self.bm25_retriever = BM25Retriever.from_documents(self.documents)
    self.bm25_retriever.k = 10
    self.ensemble_retriever = EnsembleRetriever(
        retrievers=[self.bm25_retriever, self.vector_store_retriever],
        weights=[0.5, 0.5],
    )
```

**ArXiv Tool (MCPManager)**
- **Purpose**: Singleton managing ArXiv MCP server lifecycle for paper search tools.
- **Key Methods**: `get_instance()`, `initialize()`, `get_arxiv_tools()`.
- **Code Snippet**:
```python
class MCPManager:
    _instance: Optional[MCPManager] = None
    _lock: asyncio.Lock = asyncio.Lock()

    async def initialize(self):
        server_config = {"arxiv": {"command": "arxiv-mcp-server", "args": [], "transport": "stdio"}}
        self._client = MultiServerMCPClient(server_config)
        self._tools = await self._client.get_tools()
        self._is_available = True
```

**Session Manager**
- **Purpose**: Manages chat session lifecycle and history persistence.
- **Key Methods**: `list_sessions()`, `create_session()`, `ensure_session()`.

---

### 7. Data Persistence Layer

**Database Manager (RunCRUD)**
- **Purpose**: Manages training run lifecycle and metrics persistence.
- **Key Methods**: `create()`, `update_status()`, `complete_run()`, `persist_metrics()`.
- **Code Snippet**:
```python
def complete_run(self, db: Session, run_id: int, status: str = "completed"):
    run = db.query(self.model).filter(self.model.id == run_id).first()
    updated_run = self.update(db, run_id, end_time=datetime.now(), status=status)
    self.logger.info(f"Run {run_id} updated: status={updated_run.status}")
    return updated_run

def persist_metrics(self, db: Session, run_id: int, epoch_metrics: List[Dict], ...):
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

**Metric CRUD (RunMetricCRUD)**
- **Purpose**: Manages training metrics storage with federated context support.
- **Key Methods**: `create_final_epoch_stats()`, `get_best_metric()`, `get_by_run_grouped_by_client()`.
- **Code Snippet**:
```python
def create_final_epoch_stats(self, db: Session, run_id: int, stats_dict: Dict, final_epoch: int):
    metric_mapping = {
        "sensitivity": "final_sensitivity",
        "specificity": "final_specificity",
        "precision_cm": "final_precision_cm",
    }
    metrics_data = []
    for stat_key, metric_name in metric_mapping.items():
        if stat_key in stats_dict:
            metrics_data.append({
                "run_id": run_id,
                "metric_name": metric_name,
                "metric_value": stats_dict[stat_key],
                "step": final_epoch,
            })
    return self.bulk_create(db, metrics_data)
```

**File Manager**
- **Purpose**: Handles checkpoint saving/loading and results file operations.
- **Key Methods**: `save_checkpoint()`, `load_checkpoint()`, `list_checkpoints()`.

---

## Class Diagram Reference

### API Layer Classes

**ExperimentsRouter**
- **Purpose**: FastAPI router for centralized and federated training endpoints.
- **Dependencies**: CentralizedTrainer, ServerApp.

**InferenceRouter**
- **Purpose**: FastAPI router for pneumonia detection inference endpoints.
- **Dependencies**: InferenceService.

**ChatRouter**
- **Purpose**: FastAPI router assembly combining chat sub-routers.
- **Dependencies**: SessionManager, ChatAgent.

**WebSocketManager**
- **Purpose**: Singleton managing WebSocket connections for real-time broadcasting.

---

### Training Control Classes

**CentralizedTrainer**
- **Purpose**: Orchestrates complete centralized training workflow.
- **Key Methods**: `train()`, `_load_config()`.
- **Inheritance**: None (composition-based).

**ServerApp**
- **Purpose**: Flower server orchestration for federated learning.
- **Pattern**: Decorator-based (`@app.main()`).
- **Key Methods**: `main()`, `lifespan()`.

**ClientApp**
- **Purpose**: Client-side training and evaluation in federated learning.
- **Pattern**: Decorator-based (`@app.train()`, `@app.evaluate()`).
- **Key Methods**: `train()`, `evaluate()`.

**CustomStrategy (ConfigurableFedAvg)**
- **Purpose**: Custom FedAvg strategy with WebSocket and DB integration.
- **Inheritance**: `FedAvg` (Flower).
- **Key Methods**: `configure_train()`, `aggregate_train()`, `aggregate_evaluate()`.

---

### Deep Learning Classes

**LitResNetEnhanced**
- **Purpose**: Enhanced PyTorch Lightning wrapper for ResNet50 pneumonia detection.
- **Inheritance**: `pl.LightningModule`.
- **Dependencies**: ResNetWithCustomHead, StepLogic, MetricsHandler.

**XRayDataModule**
- **Purpose**: Lightning DataModule for X-ray dataset orchestration.
- **Inheritance**: `pl.LightningDataModule`.
- **Dependencies**: TransformBuilder, CustomImageDataset.

**CustomImageDataset**
- **Purpose**: PyTorch Dataset for loading X-ray images with validation.
- **Inheritance**: `torch.utils.data.Dataset`.
- **Key Methods**: `__getitem__()`, `__len__()`, `validate_all_images()`.

**TrainingCallbacks**
- **Purpose**: Collection of PyTorch Lightning callbacks for training control.
- **Inheritance**: `pl.Callback`.
- **Callbacks**: HighestValRecallCallback, BatchMetricsCallback, GradientMonitorCallback, ProgressiveUnfreezeCallback.

---

### Inference Classes

**InferenceService**
- **Purpose**: Unified facade for pneumonia detection inference operations.
- **Dependencies**: InferenceEngine, ImageValidator, ImageProcessor.

**GradCAM**
- **Purpose**: Generates attention heatmaps for model predictions.

---

### Agentic AI Classes

**ChatAgent (ArxivAugmentedEngine)**
- **Purpose**: Research agent combining RAG and ArXiv search.
- **Inheritance**: BaseAgent.
- **Dependencies**: ChatGoogleGenerativeAI, QueryEngine, ChatHistoryManager.

**RAGPipeline (QueryEngine)**
- **Purpose**: Hybrid retrieval with BM25 and PGVector.
- **Dependencies**: EnsembleRetriever, BM25Retriever, PGVector, HuggingFaceEmbeddings.

**ArXivTool (MCPManager)**
- **Purpose**: Singleton managing ArXiv MCP server lifecycle.
- **Pattern**: Singleton with async lock.
- **Dependencies**: MultiServerMCPClient.

**SessionManager**
- **Purpose**: Manages chat session lifecycle and history.
- **Pattern**: Singleton.

---

### Persistence Classes

**RunCRUD**
- **Purpose**: Manages training run lifecycle and metrics persistence.
- **Inheritance**: BaseCRUD[Run].
- **Key Methods**: `complete_run()`, `persist_metrics()`, `batch_get_final_metrics()`.

**RunMetricCRUD**
- **Purpose**: Manages training metrics storage.
- **Inheritance**: BaseCRUD[RunMetric].
- **Key Methods**: `create_final_epoch_stats()`, `get_best_metric()`, `get_by_run_grouped_by_client()`.

**FileManager**
- **Purpose**: Handles checkpoint and results file operations.

**BaseCRUD**
- **Purpose**: Generic CRUD template using SQLAlchemy.
- **Key Methods**: `create()`, `get()`, `update()`, `delete()`, `bulk_create()`.

---

### Database Entity Classes

**Run**
- **Purpose**: SQLAlchemy entity for training runs.
- **Table**: `runs`.
- **Key Fields**: `id`, `training_mode`, `status`, `start_time`, `end_time`.
- **Relationships**: metrics (one-to-many), clients (one-to-many).

**RunMetric**
- **Purpose**: SQLAlchemy entity for training metrics.
- **Table**: `run_metrics`.
- **Key Fields**: `id`, `run_id`, `client_id`, `round_id`, `metric_name`, `metric_value`, `step`.

**ChatSession**
- **Purpose**: SQLAlchemy entity for chat sessions.
- **Table**: `chat_sessions`.
- **Key Fields**: `id`, `title`, `created_at`, `messages`.

**Message**
- **Purpose**: SQLAlchemy entity for chat messages.
- **Table**: `messages`.
- **Key Fields**: `id`, `session_id`, `content`, `role`, `timestamp`.

---

## Component Interaction Flow

```
CENTRALIZED MODE:
Dashboard → Training API → CentralizedTrainer → LitResNet/XRayDataModule → PostgreSQL

FEDERATED MODE:
Dashboard → Training API → ServerApp → ConfigurableFedAvg → ClientApp (×N) → PostgreSQL
                                                          ↓
                                                    LitResNet/XRayDataModule

INFERENCE:
Dashboard → Inference API → InferenceService → InferenceEngine → LitResNet → Prediction

CHAT:
Dashboard → Chat API → ChatAgent → QueryEngine (RAG) → PostgreSQL
                         ↓
                    MCPManager → ArXiv MCP Server

METRICS STREAMING:
Training/Federated → WebSocket Manager → Real-time Monitor (Dashboard)
```

---

## Key Architectural Patterns

1. **Clean Architecture**: Separation of API, Control, Domain, Boundary, and Entity layers.
2. **Strategy Pattern**: CustomStrategy extends FedAvg for federated learning customization.
3. **Singleton Pattern**: WebSocketManager, SessionManager, MCPManager ensure single instances.
4. **Facade Pattern**: InferenceService provides unified interface for inference operations.
5. **Repository Pattern**: CRUD classes abstract database operations.
6. **Delegation Pattern**: LitResNetEnhanced delegates step logic to StepLogic, metrics to MetricsHandler.
7. **Decorator Pattern**: Flower apps use decorators (`@app.main()`, `@app.train()`).
8. **Factory Pattern**: OptimizerFactory, LossFactory create training components.

---

## File Locations

| Component | Path |
|-----------|------|
| Training API | `federated_pneumonia_detection/src/api/endpoints/experiments/` |
| Inference API | `federated_pneumonia_detection/src/api/endpoints/inference/` |
| Chat API | `federated_pneumonia_detection/src/api/endpoints/chat/` |
| WebSocket Server | `federated_pneumonia_detection/src/api/endpoints/streaming/` |
| CentralizedTrainer | `federated_pneumonia_detection/src/control/dl_model/centralized_trainer.py` |
| ServerApp/ClientApp | `federated_pneumonia_detection/src/control/federated_new_version/core/` |
| LitResNetEnhanced | `federated_pneumonia_detection/src/control/dl_model/internals/model/` |
| XRayDataModule | `federated_pneumonia_detection/src/entities/xray_data_module.py` |
| InferenceService | `federated_pneumonia_detection/src/control/model_inferance/` |
| ChatAgent | `federated_pneumonia_detection/src/control/agentic_systems/multi_agent_systems/chat/` |
| RAG Pipeline | `federated_pneumonia_detection/src/control/agentic_systems/pipelines/rag/` |
| RunCRUD/RunMetricCRUD | `federated_pneumonia_detection/src/boundary/CRUD/` |
| Database Models | `federated_pneumonia_detection/src/boundary/models/` |
