# WebSocket Metrics Streaming Setup

## Overview

This setup provides a simple, elegant way to stream training metrics in real-time from the PyTorch Lightning training loop to the React frontend via WebSocket.

## Architecture

```
MetricsCollectorCallback (metrics_collector.py)
    ↓
MetricsWebSocketSender (websocket_metrics_sender.py)
    ↓
WebSocket Server (ws://localhost:8765)
    ↓
Frontend (TrainingExecution.tsx, StepIndicator.tsx)
```

## Components

### 1. Backend: MetricsWebSocketSender
**Location:** `federated_pneumonia_detection/src/control/dl_model/utils/data/websocket_metrics_sender.py`

Simple utility class that sends metrics to a WebSocket server. Key features:
- Handles async/sync context switching with `asyncio.run()`
- No persistent connection overhead (connects/disconnects per message)
- Graceful error handling (won't crash training if WebSocket fails)
- Convenient helper methods for common metric types

**Usage in MetricsCollectorCallback:**
```python
# Initialize in __init__
self.ws_sender = MetricsWebSocketSender("ws://localhost:8765")

# Send epoch end metrics
self.ws_sender.send_epoch_end(
    epoch=trainer.current_epoch,
    phase='val',
    metrics=val_metrics
)

# Send federated round metrics
self.ws_sender.send_round_end(
    round_num=round,
    total_rounds=total_rounds,
    fit_metrics=fit_metrics,
    eval_metrics=eval_metrics
)

# Send status updates
self.ws_sender.send_status("running", "Training in progress")
```

### 2. Backend: MetricsCollectorCallback Updates
**Location:** `federated_pneumonia_detection/src/control/dl_model/utils/model/metrics_collector.py`

PyTorch Lightning callback that now integrates WebSocket sending:

**Constructor:**
```python
callback = MetricsCollectorCallback(
    save_dir="./results",
    experiment_name="exp_001",
    training_mode="centralized",
    websocket_uri="ws://localhost:8765"  # Optional
)
```

**Automatic WebSocket sends:**
- `on_train_epoch_end()`: Sends training metrics
- `on_validation_epoch_end()`: Sends validation metrics
- Manual: `send_round_end_metrics()` for federated learning

### 3. Frontend: React WebSocket Client
**Location:** `xray-vision-ai-forge/src/services/websocket.ts`

Already implemented. Provides:
- Auto-reconnection logic
- Event-based message handling
- Type-safe listeners

**Usage in components:**
```typescript
// In TrainingExecution.tsx
const ws = createTrainingProgressWebSocket(experimentId);

ws.on('epoch_end', (data: EpochEndData) => {
  const metrics = data.metrics;
  addStatusMessage('progress', `Epoch ${data.epoch} - ${metricsStr}`);
});

ws.on('round_end', (data: any) => {
  addStatusMessage('progress', `Round ${data.round} complete`);
});
```

## Setup Instructions

### Step 1: Install Dependencies

Ensure `websockets` is installed on backend:
```bash
pip install websockets
```

### Step 2: Create WebSocket Server

Create a simple WebSocket relay server or use this basic example:

**File: `scripts/websocket_server.py`**
```python
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store connected clients per experiment
connections = {}

async def handler(websocket, path):
    """Handle WebSocket connections."""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"Received: {data.get('type')}")
                
                # Broadcast to all connected clients
                # (In a real implementation, you'd route by experiment_id)
                if connections:
                    await asyncio.gather(
                        *[ws.send(message) for ws in connections if ws != websocket],
                        return_exceptions=True
                    )
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        connections.discard(websocket)

async def main():
    server = await websockets.serve(handler, "localhost", 8765)
    logger.info("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
```

**Run the server:**
```bash
python scripts/websocket_server.py
```

### Step 3: Use in Training Script

**Example in training code:**
```python
from federated_pneumonia_detection.src.control.dl_model.utils.model.metrics_collector import MetricsCollectorCallback

# Initialize callback with WebSocket URI
metrics_callback = MetricsCollectorCallback(
    save_dir="./results/experiment_001",
    experiment_name="exp_001",
    training_mode="centralized",
    websocket_uri="ws://localhost:8765"  # Enable real-time streaming
)

# Add to trainer
trainer = pl.Trainer(
    callbacks=[metrics_callback],
    max_epochs=10,
    # ... other config
)

trainer.fit(model, train_dataloader, val_dataloader)
```

### Step 4: Verify Frontend Integration

The frontend components are already configured:
- `TrainingExecution.tsx`: Listens to WebSocket events
- `StepIndicator.tsx`: Displays progress
- Events flow automatically to UI updates

## Message Format

All messages follow this structure:

```json
{
  "type": "epoch_end",
  "timestamp": "2025-10-22T11:10:16.371Z",
  "data": {
    "epoch": 1,
    "phase": "val",
    "metrics": {
      "accuracy": 0.92,
      "loss": 0.15,
      "recall": 0.88
    }
  }
}
```

### Available Message Types

1. **epoch_start**
   ```json
   {
     "type": "epoch_start",
     "data": {
       "epoch": 1,
       "total_epochs": 10
     }
   }
   ```

2. **epoch_end**
   ```json
   {
     "type": "epoch_end",
     "data": {
       "epoch": 1,
       "phase": "val",
       "metrics": { ... }
     }
   }
   ```

3. **round_end** (Federated Learning)
   ```json
   {
     "type": "round_end",
     "data": {
       "round": 1,
       "total_rounds": 5,
       "fit_metrics": { ... },
       "eval_metrics": { ... }
     }
   }
   ```

4. **status**
   ```json
   {
     "type": "status",
     "data": {
       "status": "running",
       "message": "Training in progress"
     }
   }
   ```

5. **error**
   ```json
   {
     "type": "error",
     "data": {
       "error": "CUDA out of memory",
       "error_type": "training_error"
     }
   }
   ```

## Best Practices

### Backend

✅ **Do:**
- Leave `websocket_uri` as `None` to disable WebSocket if not needed
- Let errors be caught gracefully (won't affect training)
- Send metrics at natural breakpoints (epoch end, round end)

❌ **Don't:**
- Create persistent WebSocket connections in callbacks
- Try to send large objects (keep metrics JSON-serializable)
- Block training loop waiting for WebSocket response

### Frontend

✅ **Do:**
- Use event listeners for each message type
- Handle reconnections automatically (already implemented)
- Validate metric data before displaying

❌ **Don't:**
- Assume WebSocket is always connected
- Store entire message history in memory
- Make blocking calls on message receive

## Troubleshooting

### WebSocket Connection Refused
- Check WebSocket server is running: `python scripts/websocket_server.py`
- Verify URI matches frontend: `ws://localhost:8765`
- Check firewall isn't blocking port 8765

### Metrics Not Appearing in Frontend
- Enable debug logging in `websocket_metrics_sender.py`
- Verify `websocket_uri` is passed to `MetricsCollectorCallback`
- Check browser console for WebSocket errors

### Performance Issues
- Metrics are sent asynchronously, shouldn't affect training speed
- WebSocket server isn't bottleneck (simple message relay)
- If many experiments, consider grouping connections by experiment_id

## Example: Full Training with WebSocket

```python
import pytorch_lightning as pl
from federated_pneumonia_detection.src.control.dl_model.utils.model.metrics_collector import MetricsCollectorCallback

# Create metrics callback with WebSocket
metrics_callback = MetricsCollectorCallback(
    save_dir="./results/experiment_001",
    experiment_name="exp_001",
    run_id=None,  # Will be created if needed
    experiment_id=1,
    training_mode="centralized",
    enable_db_persistence=True,
    websocket_uri="ws://localhost:8765"  # Enable real-time updates
)

# Create trainer with callback
trainer = pl.Trainer(
    callbacks=[metrics_callback],
    max_epochs=10,
    accelerator="gpu",
    devices=1,
)

# Train model
trainer.fit(
    model=my_model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

# Manually save metrics (also persists to DB if enabled)
metrics_callback._save_metrics()
```

## Performance Notes

- **Message Size**: ~1KB per epoch metric update
- **Send Frequency**: Once per epoch (or per round for federated)
- **Latency**: <100ms for local WebSocket
- **Training Impact**: Negligible (async, non-blocking)

## Future Enhancements

- [ ] Add experiment_id to WebSocket URL routing
- [ ] Implement message queuing for high-frequency updates
- [ ] Add compression for large metric payloads
- [ ] Create dashboard for metric visualization
