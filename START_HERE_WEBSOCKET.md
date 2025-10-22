# 🚀 WebSocket Integration - START HERE

## ✅ What's Been Done

Your WebSocket metrics streaming system is **fully integrated and ready to use**.

Backend training metrics now stream to your frontend in real-time with **zero FastAPI complexity**.

## 🎯 In 30 Seconds

### 1. Install dependency
```bash
pip install websockets
```

### 2. Start FastAPI (WebSocket auto-starts!)
```bash
uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001
```

### 3. Use in training
```python
metrics_callback = MetricsCollectorCallback(
    save_dir="./results",
    experiment_name="exp_001",
    websocket_uri="ws://localhost:8765"  # ← Add this line
)
```

### 4. Run training
Frontend automatically displays real-time metrics!

---

## 📚 Documentation Structure

Choose based on your need:

| Need | File | Time |
|------|------|------|
| Just make it work | `WEBSOCKET_QUICKSTART.md` | 5 min |
| Understand the setup | `WEBSOCKET_INTEGRATION_COMPLETE.md` | 15 min |
| See API examples | `WEBSOCKET_MESSAGE_REFERENCE.md` | 10 min |
| Full deep dive | `documentation/WEBSOCKET_METRICS_SETUP.md` | 30 min |
| Verify everything | `WEBSOCKET_VERIFICATION.md` | 5 min |
| Find documentation | `WEBSOCKET_INDEX.md` | 2 min |

---

## 🔄 What Happens When You Train

```
1. Backend: Training loop runs
2. Backend: Metrics collected (on_train_epoch_end)
3. Backend: Metrics sent to ws://localhost:8765 (async, non-blocking)
4. WebSocket Server: Receives metrics
5. WebSocket Server: Broadcasts to all clients
6. Frontend: React component receives message
7. Frontend: Updates UI with real-time metrics
8. User: Sees training progress live!
```

All automatic. No additional code needed.

---

## 🧪 Test It Now

### Quick Test (2 minutes)

```bash
# Terminal 1: Start FastAPI (WebSocket auto-starts!)
uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001

# Terminal 2: Test backend components
python -c "from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import MetricsWebSocketSender; print('✓ Backend ready')"

# Terminal 3: Send test metrics (with Python)
python -c "
from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import MetricsWebSocketSender
sender = MetricsWebSocketSender('ws://localhost:8765')
sender.send_epoch_end(epoch=1, phase='val', metrics={'accuracy': 0.92, 'loss': 0.23})
print('✓ Metrics sent')
"
```

### Frontend Test (1 minute)

1. Start FastAPI: `uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001`
2. Check FastAPI logs for: `✓ WebSocket metrics server is running`
3. Open browser DevTools: `F12`
4. Go to Network tab → Filter by "WS"
5. Run the test metrics command above
6. Should see WebSocket connection and messages

---

## 📊 Features You Get

✅ **Real-time Metrics**: Epoch accuracy, loss, recall, etc.  
✅ **Live Progress**: Visual progress bar updates  
✅ **Training Status**: Running/Completed/Error notifications  
✅ **Federated Support**: Round metrics for distributed learning  
✅ **Error Handling**: Graceful failures (won't crash training)  
✅ **No Performance Hit**: Async, non-blocking sending  
✅ **Easy Integration**: Just add one parameter  
✅ **Simple Setup**: No FastAPI, no complex routing  

---

## 🎓 Example: Full Training Code

```python
import pytorch_lightning as pl
from federated_pneumonia_detection.src.control.dl_model.utils.model.metrics_collector import MetricsCollectorCallback

# Create metrics callback with WebSocket
metrics_callback = MetricsCollectorCallback(
    save_dir="./results/exp_001",
    experiment_name="exp_001",
    training_mode="centralized",
    enable_db_persistence=True,
    websocket_uri="ws://localhost:8765"  # Enable real-time updates!
)

# Create trainer
trainer = pl.Trainer(
    callbacks=[metrics_callback],
    max_epochs=10,
    accelerator="gpu",
    devices=1,
)

# Train - metrics stream to frontend automatically
trainer.fit(model, train_loader, val_loader)

# Save final metrics
metrics_callback._save_metrics()
```

---

## 🔍 What the Frontend Receives

### Train Epoch Complete
```
Epoch 1 [train] - train_loss: 0.4560, train_accuracy: 0.8700, learning_rate: 0.0010
```

### Validation Epoch Complete
```
Epoch 1 [val] - val_loss: 0.2340, val_accuracy: 0.9200, val_recall: 0.8800
```

### Federated Round Complete
```
Round 1 complete - Global accuracy: 88.00%
```

### Training Finished
```
Training completed successfully!
[Progress bar fills to 100%]
[Button changes to "View Results"]
```

---

## ⚙️ Optional: Custom Configuration

### Use Different WebSocket Port
```python
# Backend
metrics_callback = MetricsCollectorCallback(
    websocket_uri="ws://localhost:9000"  # Custom port
)

# Also update scripts/websocket_server.py line: port = 9000
```

### Disable WebSocket
```python
# Just don't pass websocket_uri
metrics_callback = MetricsCollectorCallback(
    save_dir="./results"
    # websocket_uri not provided = disabled
)
```

---

## 🚨 Troubleshooting

### "Connection refused"
→ Start WebSocket server: `python scripts/websocket_server.py`

### "No metrics appearing"
→ Check browser console (F12): should show WebSocket messages

### "Training seems slower"
→ It shouldn't be - metrics are async. Check if WebSocket server is slow.

### "TypeScript errors"
→ All types already support our message format. No changes needed.

---

## 📞 Where to Get Help

- **Quick answer**: Check `WEBSOCKET_QUICKSTART.md`
- **How it works**: Read `WEBSOCKET_INTEGRATION_COMPLETE.md`
- **API reference**: See `WEBSOCKET_MESSAGE_REFERENCE.md`
- **Deep technical**: Study `documentation/WEBSOCKET_METRICS_SETUP.md`
- **Is it correct**: Use `WEBSOCKET_VERIFICATION.md`
- **Which doc to read**: Check `WEBSOCKET_INDEX.md`

---

## ✨ That's It!

Your WebSocket integration is complete and **auto-starts with FastAPI**.

**To use:**
1. `pip install websockets`
2. Start FastAPI: `uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001`
3. Add `websocket_uri="ws://localhost:8765"` to MetricsCollectorCallback
4. Train normally - metrics appear in frontend automatically

**WebSocket server starts automatically with FastAPI. No manual setup needed!**

Questions? See the documentation files above. Everything is explained! 🚀

See `documentation/WEBSOCKET_AUTO_START.md` for auto-start details.
