# Centralized Training - Metrics Collection Flow

**API**: `POST /api/experiments/centralized/train`
**Entry Point**: `centralized_tasks.py:11` → `CentralizedTrainer.train()`

---

## Step 1: Initialize Training Run & Callbacks

**Files**:
- `centralized_tasks.py` (lines 11-65)
- `centralized_trainer.py` (lines 25-150)
- `db_operations.py` (lines 14-50)
- `training_callbacks.py` (lines 79-265)

```mermaid
sequenceDiagram
    participant API as centralized_tasks.py
    participant Trainer as CentralizedTrainer
    participant DB as db_operations.py
    participant CB as training_callbacks.py
    participant RunCRUD as run_crud
    participant WS as WebSocketSender

    API->>Trainer: run_centralized_training_task()
    Note right of API: lines 11-65

    Trainer->>DB: create_training_run()
    Note right of DB: lines 14-50

    DB->>RunCRUD: create(source_path, experiment_name, training_mode="centralized")
    Note right of RunCRUD: boundary/CRUD/run.py:30-36

    RunCRUD-->>DB: run_id
    DB-->>Trainer: run_id

    Trainer->>CB: prepare_trainer_and_callbacks_pl(run_id, ws_sender)
    Note right of CB: lines 79-265

    CB->>CB: Create callback chain (8 callbacks)
    Note right of CB: ModelCheckpoint, EarlyStopping,<br/>MetricsCollector, BatchMetrics,<br/>GradientMonitor, etc.

    CB-->>Trainer: (trainer, callbacks, metrics_collector)

    Trainer->>WS: send_metrics("training_start")
    Note right of WS: {run_id, experiment_name,<br/>max_epochs, training_mode}
```

**Key Code**:
```python
# db_operations.py lines 14-50
def create_training_run(source_path, experiment_name, logger):
    db = get_session()
    try:
        run = run_crud.create(
            db,
            source_path=source_path,
            experiment_name=experiment_name,
            training_mode="centralized",
            status="in_progress",
            start_time=datetime.now(),
        )
        db.commit()
        return run.id
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()
```

---

## Step 2: Training Loop - Epoch & Batch Metrics Collection

**Files**:
- `centralized_trainer.py` (lines 100-126)
- `metrics.py` (MetricsCollectorCallback, lines 164-216)
- `batch_metrics.py` (BatchMetricsCallback, lines 45-144)
- `gradient_monitor.py` (GradientMonitorCallback, lines 35-98)

```mermaid
sequenceDiagram
    participant PLTrainer as PyTorch Lightning Trainer
    participant MetricsCB as MetricsCollectorCallback
    participant BatchCB as BatchMetricsCallback
    participant GradCB as GradientMonitorCallback
    participant WS as WebSocketSender

    loop Each Epoch
        PLTrainer->>PLTrainer: Training batches

        loop Every Nth batch (sample_interval=10)
            PLTrainer->>BatchCB: on_train_batch_end()
            Note right of BatchCB: batch_metrics.py:45-144

            BatchCB->>BatchCB: Extract batch loss, accuracy, recall, F1
            BatchCB->>WS: send_metrics("batch_metrics")
            Note right of WS: {step, batch_idx, loss,<br/>accuracy, recall, f1, epoch}
        end

        loop Every Nth step (sample_interval=20)
            PLTrainer->>GradCB: on_before_optimizer_step()
            Note right of GradCB: gradient_monitor.py:35-98

            GradCB->>GradCB: Compute layer gradient norms
            GradCB->>WS: send_metrics("gradient_stats")
            Note right of WS: {step, total_norm, layer_norms,<br/>max_norm, min_norm}
        end

        PLTrainer->>MetricsCB: on_train_epoch_end()
        Note right of MetricsCB: metrics.py:164-176

        MetricsCB->>MetricsCB: _extract_metrics(stage="train")
        Note right of MetricsCB: Store train_loss, train_acc,<br/>train_f1, train_recall

        PLTrainer->>MetricsCB: on_validation_epoch_end()
        Note right of MetricsCB: metrics.py:178-216

        MetricsCB->>MetricsCB: _extract_metrics(stage="val")
        Note right of MetricsCB: Extract val_loss, val_acc,<br/>val_precision, val_recall,<br/>val_f1, val_auroc

        MetricsCB->>MetricsCB: Update best metrics
        Note right of MetricsCB: Track best_val_recall,<br/>best_val_loss

        MetricsCB->>WS: send_metrics("epoch_end")
        Note right of WS: {epoch, phase="val",<br/>metrics, timestamp}
    end
```

**Key Code**:
```python
# metrics.py lines 178-216 (on_validation_epoch_end)
def on_validation_epoch_end(self, trainer, pl_module):
    if trainer.sanity_checking:
        return

    val_metrics = self._extract_metrics(trainer, pl_module, "val")

    # Update epoch metrics with validation data
    if self.epoch_metrics:
        self.epoch_metrics[-1].update(val_metrics)

    # Track best metrics
    if "val_recall" in val_metrics:
        if val_metrics["val_recall"] > self.best_metrics.get("val_recall", 0):
            self.best_metrics["val_recall"] = val_metrics["val_recall"]

    # Broadcast to frontend
    self.ws_sender.send_metrics(
        {
            "epoch": trainer.current_epoch,
            "phase": "val",
            "metrics": val_metrics,
        },
        "epoch_end",
    )
```

---

## Step 3: Training Completion - DB Persistence & File Export

**Files**:
- `metrics.py` (lines 218-268)
- `metrics_file_persister.py` (lines 27-53)
- `run.py` (RunCRUD.persist_metrics, lines 234-280)
- `run_metric.py` (RunMetricCRUD.bulk_create, inherited from base.py)
- `db_operations.py` (complete_training_run, lines 66-88)

```mermaid
sequenceDiagram
    participant PLTrainer as PyTorch Lightning Trainer
    participant MetricsCB as MetricsCollectorCallback
    participant FilePersister as MetricsFilePersister
    participant RunCRUD as run_crud
    participant RunMetricCRUD as run_metric_crud
    participant DB as Database
    participant WS as WebSocketSender
    participant DBOps as db_operations.py

    PLTrainer->>MetricsCB: on_fit_end()
    Note right of MetricsCB: metrics.py:218-268

    MetricsCB->>MetricsCB: Record training_end_time

    MetricsCB->>FilePersister: save_metrics(epoch_metrics, metadata)
    Note right of FilePersister: lines 27-53

    FilePersister->>FilePersister: df.to_csv(experiment_name.csv)
    FilePersister->>FilePersister: json.dump(metadata.json)
    FilePersister-->>MetricsCB: Files saved

    MetricsCB->>MetricsCB: persist_to_database()
    Note right of MetricsCB: metrics.py:436-498

    MetricsCB->>RunCRUD: persist_metrics(run_id, epoch_metrics, federated_context=None)
    Note right of RunCRUD: run.py:234-280

    RunCRUD->>RunCRUD: _transform_epoch_to_metrics()
    Note right of RunCRUD: lines 489-535<br/>Transform Dict → List[RunMetric]

    RunCRUD->>RunMetricCRUD: bulk_create(metrics_list)
    Note right of RunMetricCRUD: Inherited from base.py:109-118

    RunMetricCRUD->>DB: db.add_all(metrics_list)
    RunMetricCRUD->>DB: db.flush()
    DB-->>RunMetricCRUD: Persisted

    RunCRUD->>DB: db.commit()
    RunCRUD-->>MetricsCB: Persistence complete

    MetricsCB->>WS: send_metrics("training_end")
    Note right of WS: {run_id, status="completed",<br/>best_epoch, best_val_recall}

    PLTrainer-->>DBOps: Training finished

    DBOps->>RunCRUD: complete_run(run_id, status="completed")
    Note right of RunCRUD: run.py:107-147

    RunCRUD->>DB: UPDATE runs SET status='completed', end_time=NOW()
    RunCRUD->>DB: Compute final confusion matrix stats
    DB-->>DBOps: Run completed
```

**Key Code**:
```python
# run.py lines 234-280 (persist_metrics)
def persist_metrics(self, db, run_id, epoch_metrics, federated_context=None):
    metrics_to_create = []

    for epoch_data in epoch_metrics:
        epoch = epoch_data.get("epoch", 0)
        client_id, round_id = self._resolve_federated_context(federated_context)

        # Transform epoch dict to metric rows
        metrics_to_create.extend(
            self._transform_epoch_to_metrics(
                run_id=run_id,
                epoch=epoch,
                epoch_data=epoch_data,
                client_id=client_id,
                round_id=round_id,
            )
        )

    # Bulk insert for performance
    run_metric_crud.bulk_create(db, metrics_to_create)
    logger.info(f"Persisted {len(metrics_to_create)} metrics for run {run_id}")
```

---

## Step 4: Results Collection & Final Stats

**Files**:
- `results.py` (lines 15-154)
- `db_operations.py` (lines 66-88)
- `run.py` (complete_run, lines 107-147)

```mermaid
sequenceDiagram
    participant Trainer as CentralizedTrainer
    participant Results as results.py
    participant DBOps as db_operations.py
    participant RunCRUD as run_crud
    participant DB as Database

    Trainer->>Results: collect_training_results()
    Note right of Results: lines 15-63

    Results->>Results: _extract_checkpoint_info()
    Note right of Results: Best model path + score

    Results->>Results: _extract_trainer_state()
    Note right of Results: Trainer stage value

    Results->>Results: _collect_metrics_data()
    Note right of Results: Metrics history + metadata

    Results->>Results: _save_results_to_file()
    Note right of Results: JSON export to results/

    Results-->>Trainer: results_dict

    Trainer->>DBOps: complete_training_run(run_id)
    Note right of DBOps: lines 66-88

    DBOps->>RunCRUD: complete_run(run_id, status="completed")
    Note right of RunCRUD: run.py:107-147

    RunCRUD->>DB: UPDATE runs SET status='completed', end_time=NOW()

    RunCRUD->>RunCRUD: Calculate final epoch stats
    Note right of RunCRUD: Confusion matrix: TP, TN, FP, FN<br/>Precision, Recall, F1, AUROC

    RunCRUD->>DB: INSERT final_* metrics
    DB-->>RunCRUD: Stats persisted

    RunCRUD-->>DBOps: Run completed
    DBOps-->>Trainer: Success
```

**Key Code**:
```python
# results.py lines 15-63
def collect_training_results(trainer, model, metrics_collector, logs_dir, checkpoint_dir, logger, run_id=None):
    results = {
        "checkpoint": _extract_checkpoint_info(trainer),
        "trainer_state": _extract_trainer_state(trainer),
        "metrics_history": metrics_collector.epoch_metrics,
        "metadata": {
            "total_epochs": trainer.current_epoch + 1,
            "best_val_recall": metrics_collector.best_metrics.get("val_recall"),
            "best_val_loss": metrics_collector.best_metrics.get("val_loss"),
            "training_time_seconds": (
                metrics_collector.training_end_time - metrics_collector.training_start_time
            ).total_seconds(),
            "run_id": run_id,
        },
    }

    _save_results_to_file(results, logs_dir)
    return results
```

---

## File Reference

| Layer | File | Key Lines |
|-------|------|-----------|
| **API Entry** | `centralized_tasks.py` | 11-65 |
| **Trainer** | `centralized_trainer.py` | 25-150 |
| **DB Setup** | `db_operations.py` | 14-50 (create), 66-88 (complete) |
| **Callbacks** | `training_callbacks.py` | 79-265 (orchestration) |
| **Metrics Collector** | `metrics.py` | 101-268 (lifecycle hooks) |
| **Batch Metrics** | `batch_metrics.py` | 45-144 |
| **Gradient Monitor** | `gradient_monitor.py` | 35-98 |
| **WebSocket** | `websocket_metrics_sender.py` | 35-106 |
| **File Persister** | `metrics_file_persister.py` | 27-53 |
| **Run CRUD** | `run.py` | 107-147 (complete), 234-280 (persist) |
| **RunMetric CRUD** | `run_metric.py` | Inherited bulk_create |
| **Results** | `results.py` | 15-154 |

---

## Metric Types Collected

| Metric Name | Source | Frequency | Storage |
|-------------|--------|-----------|---------|
| `train_loss` | MetricsCollectorCallback | Per epoch | DB + CSV |
| `train_acc`, `train_f1`, `train_recall` | MetricsCollectorCallback | Per epoch | DB + CSV |
| `val_loss`, `val_acc`, `val_precision` | MetricsCollectorCallback | Per epoch | DB + CSV |
| `val_recall`, `val_f1`, `val_auroc` | MetricsCollectorCallback | Per epoch | DB + CSV |
| `batch_loss`, `batch_accuracy` | BatchMetricsCallback | Every 10th batch | WebSocket only |
| `gradient_total_norm`, `layer_norms` | GradientMonitorCallback | Every 20th step | WebSocket only |
| `final_precision_cm`, `final_recall_cm` | FinalEpochStatsService | End of training | DB only |
| `final_f1_cm`, `final_auroc_cm` | FinalEpochStatsService | End of training | DB only |
| `cm_tp`, `cm_tn`, `cm_fp`, `cm_fn` | FinalEpochStatsService | End of training | DB only |

---

## WebSocket Event Types

| Event Type | Trigger | Payload |
|------------|---------|---------|
| `training_start` | on_train_start | {run_id, experiment_name, max_epochs, training_mode} |
| `epoch_end` | on_validation_epoch_end | {epoch, phase, metrics, timestamp} |
| `batch_metrics` | on_train_batch_end (sampled) | {step, batch_idx, loss, accuracy, recall, f1} |
| `gradient_stats` | on_before_optimizer_step (sampled) | {step, total_norm, layer_norms, max/min_norm} |
| `training_end` | on_fit_end | {run_id, status, best_epoch, best_val_recall} |
| `early_stopping` | EarlyStopping callback | {epoch, best_metric_value, patience} |
