# WebSocket Integration Guide

This guide explains how to switch from file-based progress logging to WebSocket broadcasting for real-time frontend updates.

## Current Setup (File-based)

Your training currently logs to JSON files in `logs/progress/`. This is **multiprocessing-safe** and works on Windows without pickling issues.

**Current flow:**
```
Training → ProgressLogger → JSON File → Frontend polls file
```

## WebSocket Setup (Real-time)

When your frontend is ready, switch to WebSocket broadcasting for real-time updates.

**WebSocket flow:**
```
Training → WebSocketProgressLogger → WebSocket → Frontend receives live updates
```

---

## How to Switch to WebSocket Mode

### Step 1: Import WebSocket Logger

In your metrics collector files, change the import:

**Before (File-based):**
```python
from federated_pneumonia_detection.src.utils.loggers.progress_logger import ProgressLogger
```

**After (WebSocket):**
```python
from federated_pneumonia_detection.src.utils.loggers.websocket_progress_logger import WebSocketProgressLogger
```

### Step 2: Update MetricsCollectorCallback

In `metrics_collector.py`, modify the initialization:

**Before:**
```python
self.progress_logger = ProgressLogger(
    log_dir=progress_log_dir,
    experiment_name=experiment_name,
    mode=training_mode
)
```

**After:**
```python
# websocket_manager is passed to the callback during initialization
self.progress_logger = WebSocketProgressLogger(
    websocket_manager=websocket_manager,  # Your ConnectionManager instance
    log_dir=progress_log_dir,
    experiment_name=experiment_name,
    mode=training_mode,
    enable_file_logging=True,  # Keep file backup (recommended)
    broadcast_interval=0.5      # Throttle to 2 updates/second
)
```

### Step 3: Update FederatedMetricsCollector

Similarly, in `federated_metrics_collector.py`:

**Before:**
```python
from federated_pneumonia_detection.src.utils.loggers.progress_logger import FederatedProgressLogger

self.progress_logger = FederatedProgressLogger(
    client_id=client_id,
    log_dir=progress_log_dir,
    experiment_name=experiment_name
)
```

**After:**
```python
from federated_pneumonia_detection.src.utils.loggers.websocket_progress_logger import WebSocketFederatedProgressLogger

self.progress_logger = WebSocketFederatedProgressLogger(
    client_id=client_id,
    websocket_manager=websocket_manager,
    log_dir=progress_log_dir,
    experiment_name=experiment_name,
    enable_file_logging=True,
    broadcast_interval=0.5
)
```

### Step 4: Pass WebSocket Manager

Update your callback initialization to pass the WebSocket manager:

**In `training_callbacks.py` (or wherever you create callbacks):**
```python
def prepare_trainer_and_callbacks_pl(
    # ... existing parameters ...
    websocket_manager: Optional[ConnectionManager] = None,
):
    # ... existing code ...

    metrics_callback = MetricsCollectorCallback(
        save_dir=metrics_dir,
        experiment_name=experiment_name,
        run_id=run_id,
        enable_progress_logging=True,
        progress_log_dir="logs/progress",
        websocket_manager=websocket_manager  # Pass it here
    )
```

### Step 5: Update API Endpoint

In your FastAPI endpoint that starts training:

**Example: `centralized_endpoints.py`**
```python
from federated_pneumonia_detection.src.utils.loggers.webocket_logger import ConnectionManager

# Create WebSocket manager (singleton, shared across requests)
websocket_manager = ConnectionManager()

@router.post("/train")
async def start_training(request: TrainingRequest):
    # Create trainer with WebSocket manager
    trainer = CentralizedTrainer(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        websocket_manager=websocket_manager  # Pass to trainer
    )

    # Run training
    results = trainer.train(...)

    return results
```

---

## Message Format

The WebSocket broadcasts JSON messages in this format:

### Epoch Start
```json
{
  "type": "epoch_start",
  "epoch": 1,
  "total_epochs": 10,
  "timestamp": "2025-10-20T09:32:46.123456"
}
```

### Epoch End (Training)
```json
{
  "type": "epoch_end",
  "epoch": 1,
  "phase": "train",
  "metrics": {
    "train_loss": 0.4523,
    "learning_rate": 0.001
  },
  "timestamp": "2025-10-20T09:33:12.789012"
}
```

### Epoch End (Validation)
```json
{
  "type": "epoch_end",
  "epoch": 1,
  "phase": "val",
  "metrics": {
    "val_loss": 0.3821,
    "val_accuracy": 0.8234,
    "val_recall": 0.7654,
    "val_precision": 0.8123
  },
  "timestamp": "2025-10-20T09:33:45.456789"
}
```

### Training Complete
```json
{
  "type": "training_complete",
  "final_metrics": {
    "total_epochs": 10,
    "best_epoch": 7,
    "best_val_recall": 0.8934,
    "best_val_loss": 0.2156,
    "duration_seconds": 1234.56
  },
  "timestamp": "2025-10-20T09:45:30.123456"
}
```

### Federated Round Start
```json
{
  "type": "round_start",
  "round": 1,
  "total_rounds": 5,
  "client_id": "client_0",
  "timestamp": "2025-10-20T09:32:46.123456"
}
```

### Federated Round End
```json
{
  "type": "round_end",
  "round": 1,
  "client_id": "client_0",
  "fit_metrics": {
    "train_loss": 0.4321,
    "num_samples": 1000
  },
  "eval_metrics": {
    "val_loss": 0.3987,
    "val_accuracy": 0.8456,
    "num_samples": 200
  },
  "timestamp": "2025-10-20T09:35:12.789012"
}
```

---

## Frontend Integration

### WebSocket Connection (JavaScript/TypeScript)

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/training-progress');

// Handle incoming messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'epoch_start':
      updateProgressBar(message.epoch, message.total_epochs);
      break;

    case 'epoch_end':
      if (message.phase === 'train') {
        updateTrainingMetrics(message.metrics);
      } else if (message.phase === 'val') {
        updateValidationMetrics(message.metrics);
      }
      break;

    case 'training_complete':
      showCompletionNotification(message.final_metrics);
      break;

    case 'training_error':
      showErrorAlert(message.error);
      break;

    case 'round_start':
      updateFederatedProgress(message.round, message.total_rounds);
      break;

    case 'round_end':
      updateClientMetrics(message.client_id, message.fit_metrics, message.eval_metrics);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
  // Fall back to polling JSON file
  startPollingProgressFile();
};
```

### React Example

```typescript
import { useEffect, useState } from 'react';

function TrainingMonitor() {
  const [metrics, setMetrics] = useState({});
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/training-progress');

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'epoch_end' && message.phase === 'val') {
        setMetrics(message.metrics);
        setProgress((message.epoch / message.total_epochs) * 100);
      }
    };

    return () => ws.close();
  }, []);

  return (
    <div>
      <ProgressBar value={progress} />
      <MetricsDisplay metrics={metrics} />
    </div>
  );
}
```

---

## Configuration Options

### Throttling (Prevent WebSocket Flooding)

```python
WebSocketProgressLogger(
    websocket_manager=manager,
    broadcast_interval=0.5  # Max 2 broadcasts per second
)
```

### File Backup (Recommended)

Keep file logging as backup in case WebSocket fails:

```python
WebSocketProgressLogger(
    websocket_manager=manager,
    enable_file_logging=True  # Still write to JSON file
)
```

### Disable File Logging (WebSocket Only)

```python
WebSocketProgressLogger(
    websocket_manager=manager,
    enable_file_logging=False  # Only broadcast via WebSocket
)
```

---

## Hybrid Approach (Recommended)

**Use both file and WebSocket logging:**

1. **File logging** provides:
   - Persistence (survives WebSocket disconnections)
   - Historical data
   - Debugging capability
   - Fallback for frontend polling

2. **WebSocket broadcasting** provides:
   - Real-time updates
   - Better user experience
   - No polling overhead
   - Instant feedback

**Best practice configuration:**
```python
progress_logger = WebSocketProgressLogger(
    websocket_manager=websocket_manager,
    enable_file_logging=True,      # Keep file backup
    broadcast_interval=0.5,         # Throttle to 2 updates/sec
    log_dir="logs/progress"
)
```

---

## Troubleshooting

### Issue: WebSocket Not Broadcasting

**Solution:** Check that `websocket_manager` is passed correctly:
```python
# Make sure ConnectionManager is instantiated
from federated_pneumonia_detection.src.utils.loggers.webocket_logger import ConnectionManager

manager = ConnectionManager()

# Pass to logger
logger = WebSocketProgressLogger(websocket_manager=manager)
```

### Issue: Pickling Errors with Multiprocessing

**Solution:** The WebSocket logger uses a **thread-based queue** approach that's multiprocessing-safe. The async broadcasting happens in a background thread, not in the worker processes.

### Issue: Too Many Messages

**Solution:** Increase `broadcast_interval`:
```python
WebSocketProgressLogger(
    websocket_manager=manager,
    broadcast_interval=1.0  # Only broadcast once per second
)
```

### Issue: WebSocket Disconnects

**Solution:** Enable file logging as fallback:
```python
WebSocketProgressLogger(
    websocket_manager=manager,
    enable_file_logging=True  # Frontend can poll file if WebSocket fails
)
```

---

## Migration Checklist

- [ ] Update imports in `metrics_collector.py`
- [ ] Update imports in `federated_metrics_collector.py`
- [ ] Add `websocket_manager` parameter to callback initialization
- [ ] Pass WebSocket manager from FastAPI endpoints
- [ ] Test WebSocket connection from frontend
- [ ] Verify message format matches frontend expectations
- [ ] Configure appropriate `broadcast_interval`
- [ ] Keep `enable_file_logging=True` for backup
- [ ] Test graceful degradation when WebSocket fails

---

## Summary

**Current (File-based):**
- ✅ Works now
- ✅ Multiprocessing-safe
- ✅ No pickling issues
- ❌ Requires polling

**Future (WebSocket):**
- ✅ Real-time updates
- ✅ Better UX
- ✅ Still multiprocessing-safe
- ✅ Can keep file backup
- Requires frontend WebSocket implementation

**Migration is as simple as:**
1. Change import to `WebSocketProgressLogger`
2. Pass `websocket_manager` parameter
3. Connect frontend to WebSocket endpoint

The file-based logging will continue to work as a backup!
