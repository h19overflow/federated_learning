# Federated Learning Debug Guide

## Problem: "Received 0 results and 3 failures" Error

When running `run_federated_training.py`, the Flower server reports:
```
aggregate_fit: received 0 results and 3 failures
aggregate_evaluate: received 0 results and 3 failures
```

## Root Causes and Solutions

### 1. **Windows Multiprocessing Issue with num_workers > 0** ✅ FIXED

**Problem:** On Windows, PyTorch's DataLoader with `num_workers > 0` requires scripts to be wrapped in `if __name__ == '__main__':` guard. When Flower's Ray actors try to spawn worker processes, they fail due to multiprocessing bootstrap issues.

**Error Message:**
```
RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
```

**Solution:** Set `num_workers=0` in DataLoaders for federated learning simulations.

**File Modified:** `federated_pneumonia_detection/src/control/federated_learning/data/client_data.py`

**Change Made:**
```python
# Before (lines 110, 119):
num_workers=self.config.num_workers,  # Was: 10

# After:
num_workers=0,  # Set to 0 for federated learning compatibility
```

**Why This Works:**
- Flower's simulation runs clients via Ray actors on Windows
- Ray's worker spawning conflicts with PyTorch's multiprocessing when num_workers > 0
- Setting num_workers=0 disables multiprocessing in DataLoader
- With 256x256 images and modern hardware, data loading speed isn't bottlenecked by single-threaded loading
- This is the recommended approach for federated learning simulations on Windows

### 2. **Config Issue: batch_size Too Large**

**Config Parameter:** `batch_size: 512` in `default_config.yaml`

**Concern:** With 256x256 RGB images and ResNet50, batch size 512 requires significant VRAM. If you encounter OOM errors with num_workers=0, reduce batch size to 32-64.

**Verification:** Check if getting CUDA OOM errors in logs.

### 3. **Flower Framework Limitations**

Flower 1.22.0's `start_simulation()` uses Ray actors to run clients. The current implementation works but is deprecated in favor of using `flwr run` CLI command.

**Status:** Currently working after the num_workers fix above.

## Testing the Fix

To verify the fix works:

```bash
python run_federated_training.py
```

Look for output like:
```
[ROUND 1]
configure_fit: strategy sampled 3 clients (out of 5)
aggregate_fit: received 3 results and 0 failures
```

If you see "0 results and N failures", there's still an issue. Check:
1. Logs in `results/federated/logs/` for specific errors
2. Ensure num_workers=0 is in effect
3. Check for CUDA memory issues

## Configuration Reference

**Default config** (`default_config.yaml`):
```yaml
experiment:
  num_rounds: 15
  num_clients: 5
  clients_per_round: 3
  local_epochs: 2
  batch_size: 512          # Adjust down if OOM issues
  num_workers: 10          # NOW IGNORED in FL, always 0
  pin_memory: true
```

## Monitoring

Watch for these indicators of success:
1. Clients are created: Check for `Creating DataLoaders` messages
2. Training progresses: Check for loss values in each round
3. Parameters are updated: Check history for losses_distributed

## Environment Info

- **OS:** Windows
- **Flower Version:** 1.22.0
- **PyTorch:** Current stable
- **Issue Scope:** Federated learning simulations on Windows

## Next Steps If Still Failing

1. Enable DEBUG logging:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Check Ray logs in `~/ray_results/`

3. Verify CUDA/GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

4. Try reducing batch_size to 32

5. Test with local client directly (without Flower) to isolate issues
