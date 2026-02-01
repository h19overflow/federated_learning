# Federated Client Training - Metrics Collection Flow

**Entry Point**: `client_app.py:36` → `@app.train(msg, context)`
**Pattern**: Flower ClientApp with local training on IID/non-IID partitions

---

## Step 1: Client Initialization & Partition Loading

**Files**:
- `client_app.py` (lines 36-108)
- `utils.py` (`_get_partition_data`, `_prepare_partition_and_split`)
- `training_callbacks.py` (lines 79-265)

```mermaid
sequenceDiagram
    participant Flower as Flower Framework
    participant ClientApp as client_app.py
    participant Utils as utils.py
    participant Partitioner as IidPartitioner
    participant CB as training_callbacks.py

    Flower->>ClientApp: train(msg, context)
    Note right of ClientApp: lines 36-179

    ClientApp->>ClientApp: Extract client_id, round_number
    Note right of ClientApp: context.node_id, context.state.current_round

    ClientApp->>Utils: _get_partition_data(configs)
    Note right of Utils: Load full dataset + create partitioner

    Utils->>Partitioner: IidPartitioner(num_partitions=N)
    Utils-->>ClientApp: (dataset, partitioner)

    ClientApp->>ClientApp: partition_id = client_id % num_partitions

    ClientApp->>Partitioner: load_partition(partition_id)
    Partitioner-->>ClientApp: partition_df

    ClientApp->>Utils: _prepare_partition_and_split(partition_df)
    Note right of Utils: Train/val split with stratification

    Utils-->>ClientApp: (train_df, val_df)

    ClientApp->>ClientApp: Create XRayDataModule(train_df, val_df)

    ClientApp->>Utils: _build_model_components(is_federated=True, client_id, round_number, run_id)
    Note right of Utils: Create model + callbacks chain

    Utils->>CB: prepare_trainer_and_callbacks_pl(client_id, round_number, run_id)
    Note right of CB: lines 79-265<br/>Federated mode enabled

    CB-->>Utils: (callbacks, metrics_collector)
    Utils-->>ClientApp: (model, callbacks, metrics_collector)
```

**Key Code**:
```python
# client_app.py lines 36-108
@app.train()
def train(msg: Message, context: Context):
    centerlized_trainer, config = _load_trainer_and_config()

    client_id = context.node_id
    round_number = (
        context.state.current_round if hasattr(context.state, "current_round") else 0
    )

    configs = msg.content.get("config", {...})
    _, partioner = _get_partition_data(configs)
    partition_id = context.node_id % configs["num_partitions"]
    partion_df = partioner.load_partition(partition_id)

    train_df, val_df = _prepare_partition_and_split(
        partioner, partition_id, partion_df, seed=configs.get("seed", 42)
    )

    data_module = XRayDataModule(
        train_df=train_df, val_df=val_df, config=config, image_dir=configs["image_dir"]
    )

    run_id = configs.get("run_id", None)

    model, callbacks, metrics_collector = _build_model_components(
        centerlized_trainer,
        train_df,
        context,
        is_federated=True,
        client_id=client_id,
        round_number=round_number,
        run_id=run_id,
    )
```

---

## Step 2: Load Global Model & Local Training

**Files**:
- `client_app.py` (lines 110-136)
- `metrics.py` (MetricsCollectorCallback, federated mode)
- `batch_metrics.py` (with client_id tagging)

```mermaid
sequenceDiagram
    participant ClientApp as client_app.py
    participant Model as PyTorch Model
    participant PLTrainer as PyTorch Lightning Trainer
    participant MetricsCB as MetricsCollectorCallback
    participant BatchCB as BatchMetricsCallback
    participant DB as Database (Client DB)
    participant WS as WebSocketSender

    ClientApp->>ClientApp: global_state_dict = msg.content["arrays"].to_torch_state_dict()
    Note right of ClientApp: lines 110-111<br/>Receive weights from server

    ClientApp->>Model: load_state_dict(global_state_dict)
    Note right of Model: Initialize with global model

    ClientApp->>ClientApp: Log first param before training
    Note right of ClientApp: lines 113-118

    ClientApp->>PLTrainer: fit(model, data_module)
    Note right of PLTrainer: Local training starts

    PLTrainer->>MetricsCB: on_train_start()
    Note right of MetricsCB: metrics.py:101-162

    MetricsCB->>DB: _ensure_run_exists() + _ensure_client_exists()
    Note right of MetricsCB: lines 346-434<br/>Create Client entity with client_id

    DB-->>MetricsCB: client_db_id

    MetricsCB->>WS: send_metrics("training_start")
    Note right of WS: {run_id, client_id, round_number,<br/>experiment_name, training_mode="federated"}

    loop Each Local Epoch
        PLTrainer->>PLTrainer: Training batches on local partition

        loop Every Nth batch
            PLTrainer->>BatchCB: on_train_batch_end()
            BatchCB->>WS: send_metrics("batch_metrics")
            Note right of WS: {step, batch_idx, loss,<br/>client_id, round_num}
        end

        PLTrainer->>MetricsCB: on_train_epoch_end()
        MetricsCB->>MetricsCB: _extract_metrics(stage="train")

        PLTrainer->>MetricsCB: on_validation_epoch_end()
        MetricsCB->>MetricsCB: _extract_metrics(stage="val")
        MetricsCB->>WS: send_metrics("epoch_end")
        Note right of WS: {epoch, phase="val", metrics,<br/>client_id, round_number}
    end

    PLTrainer-->>ClientApp: Training complete

    ClientApp->>ClientApp: Log first param after training
    Note right of ClientApp: lines 128-136<br/>Verify parameter updates
```

**Key Code**:
```python
# client_app.py lines 110-136
global_state_dict = msg.content["arrays"].to_torch_state_dict()
model.load_state_dict(global_state_dict)

first_param_name = list(model.state_dict().keys())[0]
first_param_before = model.state_dict()[first_param_name].clone()
centerlized_trainer.logger.info(
    f"[Client Train] BEFORE training - first param '{first_param_name}' "
    f"mean: {first_param_before.mean().item():.6f}"
)

trainer.fit(model, data_module)

first_param_after = model.state_dict()[first_param_name]
centerlized_trainer.logger.info(
    f"[Client Train] AFTER training - first param '{first_param_name}' "
    f"mean: {first_param_after.mean().item():.6f}"
)
centerlized_trainer.logger.info(
    f"[Client Train] Parameter change: "
    f"{(first_param_after - first_param_before).abs().mean().item():.6f}"
)
```

---

## Step 3: Collect Results & DB Persistence (Federated Context)

**Files**:
- `client_app.py` (lines 137-179)
- `results.py` (collect_training_results)
- `utils.py` (filter_list_of_dicts)
- `metrics.py` (persist_to_database with federated_context)
- `run.py` (persist_metrics with client_id, round_id)

```mermaid
sequenceDiagram
    participant ClientApp as client_app.py
    participant Results as results.py
    participant MetricsCB as MetricsCollectorCallback
    participant RunCRUD as run_crud
    participant RunMetricCRUD as run_metric_crud
    participant DB as Database
    participant Flower as Flower Framework

    ClientApp->>Results: collect_training_results()
    Note right of Results: lines 15-63

    Results->>MetricsCB: Get metrics_history
    Results-->>ClientApp: results_dict

    ClientApp->>ClientApp: filter_list_of_dicts(metrics_history, selected_metrics)
    Note right of ClientApp: lines 149-163<br/>Extract: epoch, train_loss, train_acc,<br/>val_loss, val_acc, val_f1, val_auroc

    ClientApp->>ClientApp: Add num_examples to metrics
    Note right of ClientApp: line 165<br/>metrics_history["num-examples"] = len(train_df)

    ClientApp->>MetricsCB: persist_to_database()
    Note right of MetricsCB: metrics.py:436-498

    MetricsCB->>MetricsCB: Prepare federated_context
    Note right of MetricsCB: {<br/>  client_id: self.db_client_id,<br/>  round_number: self.current_round<br/>}

    MetricsCB->>RunCRUD: persist_metrics(run_id, epoch_metrics, federated_context)
    Note right of RunCRUD: run.py:234-280

    RunCRUD->>RunCRUD: _resolve_federated_context()
    Note right of RunCRUD: lines 431-468<br/>Extract client_id, round_id from context

    RunCRUD->>RunCRUD: _transform_epoch_to_metrics()
    Note right of RunCRUD: lines 489-535<br/>Include client_id, round_id in metrics

    RunCRUD->>RunMetricCRUD: bulk_create(metrics_list)
    Note right of RunMetricCRUD: Each metric tagged with:<br/>- run_id<br/>- client_id<br/>- round_id

    RunMetricCRUD->>DB: db.add_all(metrics_list)
    RunMetricCRUD->>DB: db.flush() + db.commit()
    DB-->>RunCRUD: Persisted

    RunCRUD-->>MetricsCB: Success
    MetricsCB-->>ClientApp: Persistence complete

    ClientApp->>ClientApp: Create Message with model updates + metrics
    Note right of ClientApp: lines 171-179<br/>ArrayRecord(model.state_dict())<br/>MetricRecord(metrics_history)

    ClientApp-->>Flower: Return Message(content, reply_to=msg)
    Note right of Flower: Send updated weights + metrics<br/>back to server
```

**Key Code**:
```python
# client_app.py lines 137-179
results = collect_training_results(
    trainer=trainer,
    model=model,
    metrics_collector=metrics_collector,
    logs_dir=config.get("output.log_dir"),
    checkpoint_dir=config.get("output.checkpoint_dir"),
    logger=centerlized_trainer.logger,
    run_id=run_id,
)

num_examples = len(train_df)

metrics_history = filter_list_of_dicts(
    results.get("metrics_history", []),
    ["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc",
     "val_precision", "val_recall", "val_f1", "val_auroc"],
)

metrics_history["num-examples"] = int(num_examples)

# Return to server
model_record = ArrayRecord(model.state_dict())
metric_record = MetricRecord(metrics_history)
content = RecordDict({
    "arrays": model_record,
    "metrics": metric_record,
})
return Message(content=content, reply_to=msg)
```

```python
# run.py lines 431-468 (_resolve_federated_context)
def _resolve_federated_context(self, db, federated_context):
    if not federated_context:
        return None, None

    client_id = federated_context.get("client_id")
    round_number = federated_context.get("round_number")

    # Get client DB ID
    client = db.query(Client).filter(Client.client_id == client_id).first()
    client_db_id = client.id if client else None

    # Get round DB ID
    round_obj = db.query(Round).filter(
        Round.round_number == round_number
    ).first()
    round_db_id = round_obj.id if round_obj else None

    return client_db_id, round_db_id
```

---

## Step 4: Client Evaluation (Optional)

**Files**:
- `client_app.py` (lines 182-295)
- `server_evaluation.py` (central evaluation function)

```mermaid
sequenceDiagram
    participant Flower as Flower Framework
    participant ClientApp as client_app.py
    participant Model as PyTorch Model
    participant PLTrainer as PyTorch Lightning Trainer
    participant Utils as utils.py

    Flower->>ClientApp: evaluate(msg, context)
    Note right of ClientApp: lines 182-295

    ClientApp->>ClientApp: Load evaluation configs
    Note right of ClientApp: csv_path, image_dir

    ClientApp->>ClientApp: prepare_dataset()
    Note right of ClientApp: Load full test set (NOT partitioned)

    ClientApp->>ClientApp: create_data_module(train_df, val_df)

    ClientApp->>Utils: _build_model_components(is_federated=False)
    Utils-->>ClientApp: (model, callbacks, metrics_collector)

    ClientApp->>Model: load_state_dict(global_state_dict)
    Note right of Model: Load global model from server

    ClientApp->>PLTrainer: test(model, val_loader)
    Note right of PLTrainer: Evaluate on local val set

    PLTrainer-->>ClientApp: result_dict

    ClientApp->>Utils: _extract_metrics_from_result(result_dict)
    Note right of Utils: Extract loss, accuracy, precision,<br/>recall, f1, auroc, confusion matrix

    Utils-->>ClientApp: (loss, acc, prec, rec, f1, auroc, cm_tp, cm_tn, cm_fp, cm_fn)

    ClientApp->>Utils: _create_metric_record_dict()
    Note right of Utils: Bundle metrics with num_examples

    Utils-->>ClientApp: metric_dict

    ClientApp->>ClientApp: Create MetricRecord(metric_dict)

    ClientApp-->>Flower: Return Message(content={metrics})
    Note right of Flower: Send evaluation metrics to server
```

**Key Code**:
```python
# client_app.py lines 260-295
results = trainer.test(model, val_loader)
result_dict = results[0] if results else {}

loss, accuracy, precision, recall, f1, auroc, cm_tp, cm_tn, cm_fp, cm_fn = (
    _extract_metrics_from_result(result_dict)
)

num_examples = len(val_df)

metric_dict = _create_metric_record_dict(
    loss, accuracy, precision, recall, f1, auroc, num_examples,
    cm_tp, cm_tn, cm_fp, cm_fn,
)

centerlized_trainer.logger.info(
    f"[Client Evaluate] Extracted metrics: loss={loss}, acc={accuracy}, "
    f"prec={precision}, rec={recall}, f1={f1}, auroc={auroc}, "
    f"cm_tp={cm_tp}, cm_tn={cm_tn}, cm_fp={cm_fp}, cm_fn={cm_fn}, "
    f"num_examples={num_examples}"
)

metric_record = MetricRecord(metric_dict)
content = RecordDict({"metrics": metric_record})
return Message(content=content, reply_to=msg)
```

---

## File Reference

| Layer | File | Key Lines |
|-------|------|-----------|
| **Client Entry** | `client_app.py` | 36-179 (train), 182-295 (evaluate) |
| **Utils** | `utils.py` | Partition, model build, metric extraction |
| **Callbacks** | `training_callbacks.py` | 79-265 (federated mode) |
| **Metrics Collector** | `metrics.py` | 101-162 (federated client init), 346-434 (DB client) |
| **Batch Metrics** | `batch_metrics.py` | 45-144 (with client_id tagging) |
| **Run CRUD** | `run.py` | 234-280 (persist with federated_context), 431-468 (resolve context) |
| **RunMetric CRUD** | `run_metric.py` | Bulk create with client_id, round_id |
| **Results** | `results.py` | 15-63 (collect training results) |

---

## Federated Context Propagation

| Component | Context Fields | Lines |
|-----------|---------------|-------|
| **ClientApp** | `client_id = context.node_id` | client_app.py:45 |
| | `round_number = context.state.current_round` | client_app.py:46-48 |
| **MetricsCollectorCallback** | `self.client_id`, `self.current_round` | metrics.py:90-91 |
| | `self.db_client_id` (from DB Client entity) | metrics.py:155-161 |
| **RunCRUD.persist_metrics** | `federated_context = {client_id, round_number}` | run.py:234-280 |
| **RunMetricCRUD** | Each metric tagged with `client_id`, `round_id` | run_metric.py (inherited) |

---

## Key Differences from Centralized

| Aspect | Centralized | Federated Client |
|--------|-------------|------------------|
| **Run Creation** | API creates run | Server creates run, passes run_id in config |
| **Client Entity** | N/A | `_ensure_client_exists()` creates Client row |
| **Metrics Context** | `federated_context=None` | `federated_context={client_id, round_number}` |
| **DB Persistence** | All metrics under `run_id` | Metrics tagged with `run_id`, `client_id`, `round_id` |
| **WebSocket Events** | `training_end` sent by trainer | NO `training_end` (server sends it) |
| **Model Return** | N/A | Returns `ArrayRecord(state_dict)` to server |
| **Num Examples** | Total dataset size | Local partition size |

---

## Metrics Returned to Server

| Metric | Purpose |
|--------|---------|
| `num-examples` | Weight for FedAvg aggregation |
| `train_loss`, `train_acc`, `train_f1` | Training performance on local data |
| `val_loss`, `val_acc`, `val_f1`, `val_auroc` | Validation performance (if applicable) |
| `epoch` | Training progress indicator |

**Aggregation**: Server uses `num-examples` to compute weighted average in `custom_strategy.py:aggregate_fit()`.
