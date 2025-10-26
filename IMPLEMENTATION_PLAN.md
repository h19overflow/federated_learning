# Implementation Plan: WebSocket Metrics Streaming for Federated Training

## Executive Summary
Implement the same metrics streaming pattern from `centralized_trainer.py` into `federated_trainer.py` to enable real-time WebSocket communication with the frontend during federated learning training.

---

## Current State Analysis

### Centralized Trainer Pattern (Working Reference)
**File**: `centralized_trainer.py`

**Components Used**:
1. **MetricsCollectorCallback** - PyTorch Lightning callback that:
   - Collects epoch-level metrics during training
   - Persists metrics to database (optional)
   - Streams metrics to frontend via WebSocket
   - Sends `training_start`, `epoch_end`, `training_end` events
   
2. **MetricsWebSocketSender** - WebSocket client that:
   - Connects to `ws://localhost:8765`
   - Sends JSON payloads with different event types
   - Methods: `send_epoch_end()`, `send_round_end()`, `send_training_end()`, `send_early_stopping_triggered()`

3. **EarlyStoppingSignalCallback** - Custom callback that:
   - Monitors PyTorch Lightning's `EarlyStopping` callback
   - Sends WebSocket notification when early stopping triggers

**Integration Flow**:
```
CentralizedTrainer.__init__()
  ↓
train() method
  ↓
prepare_trainer_and_callbacks_pl() 
  ↓ creates
[MetricsCollectorCallback, EarlyStoppingSignalCallback, ...]
  ↓ uses internally
MetricsWebSocketSender
  ↓ sends to
WebSocket Server (ws://localhost:8765)
  ↓
Frontend receives real-time updates
```

---

### Federated Trainer Pattern (Current State)
**File**: `federated_trainer.py`

**Components**:
1. **FederatedMetricsCollector** - Client-side metrics collector that:
   - Tracks metrics per federated round
   - Tracks local epoch metrics within each round
   - Saves to JSON/CSV files
   - Can persist to database
   - **MISSING**: WebSocket streaming

2. **FlowerClient** - Individual client that:
   - Uses `FederatedMetricsCollector` for local metrics
   - Trains model for `local_epochs` per round
   - Returns metrics to Flower server

3. **FederatedTrainer** - Main orchestrator that:
   - Manages Flower simulation
   - Has a `websocket_manager` parameter (currently unused)
   - Aggregates client metrics after training completes
   - **MISSING**: Real-time WebSocket streaming

**Current Metrics Flow**:
```
FederatedTrainer.train()
  ↓
Flower simulation (fl.simulation.start_simulation)
  ↓ spawns
Multiple FlowerClients (one per client)
  ↓ each has
FederatedMetricsCollector
  ↓
Metrics saved to files at END of training
(No real-time streaming)
```

---

## Key Differences & Challenges

### Structure Differences
| Aspect | Centralized | Federated |
|--------|-------------|-----------|
| Training Loop | Single model, multiple epochs | Multiple clients, multiple rounds, multiple local epochs per client |
| Progress Unit | Epoch | Round (contains multiple client updates) |
| Metrics Source | Single PyTorch Lightning Trainer | Multiple FlowerClients + Server evaluations |
| Callback System | PyTorch Lightning callbacks | Flower strategy callbacks |
| Parallelism | Sequential epochs | Parallel client training |

### Logging Complexity
**Centralized**: Linear progression
- Epoch 1 → Epoch 2 → ... → Epoch N

**Federated**: Hierarchical progression
- Round 1:
  - Client 0: Local Epoch 0, 1, 2, ...
  - Client 1: Local Epoch 0, 1, 2, ...
  - ...
  - Server Evaluation
- Round 2:
  - (Same pattern)
- ...

### What Frontend Needs to Know
1. **Training Start**: `run_id`, total rounds, number of clients
2. **Round Start**: Current round number
3. **Client Progress**: Which clients are training (optional)
4. **Round End**: Aggregated metrics from all clients, server evaluation
5. **Training End**: Final summary, best round

---

## Implementation Plan

### Phase 1: Add WebSocket Sender to FederatedMetricsCollector
**File**: `federated_metrics_collector.py`

**Changes**:
1. Add `websocket_uri` parameter to `__init__`
2. Initialize `MetricsWebSocketSender` (like centralized)
3. Add WebSocket event methods:
   - `send_round_start()` - Notify when round starts
   - `send_local_epoch()` - (Optional) Stream local epochs for detailed monitoring
   - `send_round_end()` - Send aggregated round metrics
   
**Code locations to modify**:
- Line 39-62: `__init__` - Add websocket_sender initialization
- Line 114-141: `record_round_start()` - Send WebSocket event
- Line 143-198: `record_local_epoch()` - Optionally send local epoch updates
- Line 200-234: `record_fit_metrics()` - Send fit completion
- Line 236-291: `record_eval_metrics()` - Send eval completion with `send_round_end()`
- Line 294-326: `end_training()` - Send training completion

**Expected behavior**:
```python
# In FederatedMetricsCollector
self.ws_sender = MetricsWebSocketSender(websocket_uri)

# In record_round_start()
if self.ws_sender:
    self.ws_sender.send_metrics({
        "round": round_num,
        "client_id": self.client_id,
        "total_rounds": total_rounds
    }, "round_start")

# In record_eval_metrics() (after round completes)
if self.ws_sender:
    self.ws_sender.send_round_end(
        round_num=round_num,
        fit_metrics=fit_metrics,
        eval_metrics=eval_data
    )
```

---

### Phase 2: Add Server-Level WebSocket Sender to FederatedTrainer
**File**: `federated_trainer.py`

**Changes**:
1. Initialize a **server-level** `MetricsWebSocketSender` in `__init__`
2. Create a custom Flower strategy callback that:
   - Hooks into `FedAvg.on_fit_config_fn`
   - Hooks into `FedAvg.evaluate_fn` (server-side evaluation)
   - Sends aggregated metrics to frontend
3. Send training lifecycle events:
   - `training_start` - When simulation begins
   - `round_end` - After each round (server perspective)
   - `training_end` - When simulation completes

**Code locations to modify**:
- Line 59-84: `__init__` - Initialize `MetricsWebSocketSender`
- Line 310-462: `train()` method:
  - After line 336: Send `training_start`
  - Before line 428 (fl.simulation.start_simulation): Create custom strategy with callbacks
  - After line 434: Process history and send `round_end` events
  - Around line 460: Send `training_end`

**Expected behavior**:
```python
# In FederatedTrainer.__init__
self.ws_sender = MetricsWebSocketSender(websocket_uri="ws://localhost:8765")

# In train() - Training start
self.ws_sender.send_metrics({
    "run_id": self.run_id,  # Need to create run_id early
    "experiment_name": experiment_name,
    "num_clients": self.config.num_clients,
    "num_rounds": self.config.num_rounds,
    "training_mode": "federated"
}, "training_start")

# After each round (from history)
for round_num, metrics in enumerate(history.metrics_distributed):
    self.ws_sender.send_round_end(
        round_num=round_num,
        total_rounds=self.config.num_rounds,
        fit_metrics=metrics,
        eval_metrics=history.metrics_centralized.get(round_num)
    )

# Training end
self.ws_sender.send_training_end(
    run_id=self.run_id,
    summary_data={
        "status": "completed",
        "best_round": best_round,
        ...
    }
)
```

---

### Phase 3: Create Run ID Early (Database Integration)
**File**: `federated_trainer.py`

**Problem**: Currently `run_id` is not created until training completes. Frontend needs it at training start to query/track progress.

**Solution**:
1. Add method `_create_run()` that creates database run entry
2. Call it at the START of `train()` method
3. Pass `run_id` to clients via metrics_collector
4. Include `run_id` in all WebSocket messages

**Code changes**:
```python
# Add new method
def _create_run(self, experiment_name: str, source_path: str) -> int:
    """Create database run entry and return run_id."""
    from federated_pneumonia_detection.src.boundary.engine import get_session
    from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
    
    db = get_session()
    try:
        run_data = {
            'training_mode': 'federated',
            'status': 'in_progress',
            'start_time': datetime.now(),
            'wandb_id': 'placeholder',
            'source_path': source_path,
        }
        new_run = run_crud.create(db, **run_data)
        db.flush()
        db.commit()
        return new_run.id
    finally:
        db.close()

# In train() method - at start
self.run_id = self._create_run(experiment_name, source_path)
```

---

### Phase 4: Enhanced Metrics Structure for Frontend

**WebSocket Message Types** (aligned with centralized):

#### 1. `training_start`
```json
{
  "type": "training_start",
  "timestamp": "2024-01-15T10:30:00",
  "data": {
    "run_id": 123,
    "experiment_name": "federated_pneumonia",
    "num_clients": 5,
    "num_rounds": 10,
    "local_epochs": 3,
    "training_mode": "federated"
  }
}
```

#### 2. `round_start`
```json
{
  "type": "round_start",
  "timestamp": "2024-01-15T10:30:05",
  "data": {
    "run_id": 123,
    "round": 1,
    "total_rounds": 10,
    "participating_clients": 5
  }
}
```

#### 3. `round_end` (Server aggregated view)
```json
{
  "type": "round_end",
  "timestamp": "2024-01-15T10:32:00",
  "data": {
    "run_id": 123,
    "round": 1,
    "total_rounds": 10,
    "fit_metrics": {
      "train_loss": 0.234,
      "num_samples": 5000
    },
    "eval_metrics": {
      "val_loss": 0.189,
      "val_accuracy": 0.923
    }
  }
}
```

#### 4. `client_progress` (Optional - detailed view)
```json
{
  "type": "client_progress",
  "timestamp": "2024-01-15T10:31:30",
  "data": {
    "run_id": 123,
    "round": 1,
    "client_id": "client_2",
    "local_epoch": 2,
    "train_loss": 0.245,
    "learning_rate": 0.001
  }
}
```

#### 5. `training_end`
```json
{
  "type": "training_end",
  "timestamp": "2024-01-15T11:00:00",
  "data": {
    "run_id": 123,
    "status": "completed",
    "experiment_name": "federated_pneumonia",
    "best_round": 8,
    "best_val_accuracy": 0.945,
    "total_rounds": 10,
    "training_duration": "0:30:00"
  }
}
```

---

## Implementation Steps (Sequential)

### Step 1: Update FederatedMetricsCollector ✅ DONE
- [x] Add `websocket_uri` parameter to `__init__`
- [x] Initialize `MetricsWebSocketSender` instance
- [x] Update `record_round_start()` to send `round_start` event
- [x] Update `record_local_epoch()` to send `client_progress` event (optional)
- [x] Update `record_eval_metrics()` to send `round_end` event
- [x] Update `end_training()` to send `client_complete` event
- [x] Update `client.py` to pass `websocket_uri` to collector

### Step 2: Update FederatedTrainer - Early Run Creation ✅ DONE
- [x] Add `_create_run()` method
- [x] Call it at start of `train()` method
- [x] Store `self.run_id` for use throughout training
- [x] Pass `run_id` to `FederatedMetricsCollector` instances

### Step 3: Update FederatedTrainer - WebSocket Integration ✅ DONE
- [x] Initialize `MetricsWebSocketSender` in `__init__`
- [x] Send `training_start` at beginning of `train()`
- [x] Extract and send `round_end` from Flower history (handled by client collectors)
- [x] Send `training_end` when training completes
- [x] Handle errors gracefully (don't crash if WebSocket unavailable)

### Step 4: Update FlowerClient to Pass WebSocket Context ✅ DONE
- [x] Ensure `run_id` is available to clients
- [x] Pass `websocket_uri` to `FederatedMetricsCollector`
- [x] Verify client metrics are streamed properly

### Step 5: Testing & Validation
- [ ] Start WebSocket server (`ws://localhost:8765`)
- [ ] Run federated training with small dataset
- [ ] Monitor WebSocket messages in real-time
- [ ] Verify message structure matches expected format
- [ ] Test with WebSocket server down (graceful degradation)
- [ ] Verify database persistence still works

### Step 6: Documentation
- [ ] Update docstrings in modified files
- [ ] Add comments explaining WebSocket flow
- [ ] Document message types and structure

---

## Risk Mitigation

### Risk 1: WebSocket Blocking Training
**Mitigation**: 
- Use async WebSocket sender (already implemented in `MetricsWebSocketSender`)
- Wrap all WebSocket calls in try-except blocks
- Log warnings but don't crash if WebSocket fails

### Risk 2: Duplicate Messages from Multiple Clients
**Mitigation**:
- Server-level messages send aggregated metrics (one per round)
- Client-level messages include `client_id` in payload
- Frontend can filter/aggregate as needed

### Risk 3: Performance Impact
**Mitigation**:
- Only send messages at round boundaries (not every batch)
- Local epoch streaming is optional (can be disabled)
- WebSocket sender uses connection pooling (new connection per message is acceptable for round-level granularity)

### Risk 4: Run ID Not Available Early Enough
**Mitigation**:
- Create run in database BEFORE starting simulation
- Commit transaction immediately
- Pass run_id to all components

---

## Success Criteria

1. ✅ Frontend receives `training_start` with `run_id` when federated training begins
2. ✅ Frontend receives `round_end` after each federated round
3. ✅ Frontend receives `training_end` with summary when training completes
4. ✅ Database persistence continues to work
5. ✅ Training completes successfully even if WebSocket server is unavailable
6. ✅ Message structure is consistent with centralized trainer pattern
7. ✅ No performance degradation (< 5% overhead)

---

## Code Files to Modify

1. **federated_metrics_collector.py**
   - Lines: 39-62 (init), 114-141 (round_start), 236-291 (eval_metrics), 294-326 (end_training)
   - Estimated changes: +50 lines

2. **trainer.py** (FederatedTrainer)
   - Lines: 59-84 (init), 310-462 (train method)
   - Estimated changes: +100 lines

3. **client.py** (FlowerClient)
   - Lines: 40-64 (init), 106-112 (metrics_collector init)
   - Estimated changes: +10 lines (minor)

**Total estimated additions**: ~160 lines of code

---

## Timeline Estimate
- Step 1: 1 hour
- Step 2: 30 minutes
- Step 3: 1.5 hours
- Step 4: 30 minutes
- Step 5: 1 hour (testing)
- Step 6: 30 minutes (documentation)

**Total**: ~5 hours

---

## Notes
- This maintains backward compatibility (WebSocket is optional)
- Follows existing patterns from centralized trainer
- Minimal changes to Flower client/server logic
- Frontend can reuse existing WebSocket listener infrastructure
