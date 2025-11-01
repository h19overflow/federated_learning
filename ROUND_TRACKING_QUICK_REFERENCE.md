# Round Tracking Quick Reference

## What Was Fixed

### Three Critical Issues Resolved:

1. **WebSocket Rounds (trainer.py)**
   - ❌ Was sending: rounds 0, 1 only
   - ✅ Now sends: rounds 0, 1, 2, 3 (for 3-round training)
   - **How**: Added `RoundTrackingFedAvg` strategy to send event after each round

2. **Database Round Numbers (client.py)**
   - ❌ Was storing: 0, 0, 0 for all rounds
   - ✅ Now stores: 0, 1, 2 for each round
   - **How**: Capture round number at fit() START, increment at END

3. **Metrics-to-Round Linking (federated_metrics_collector.py)**
   - ❌ Was linking: only last round to database ID
   - ✅ Now links: all rounds to their database IDs
   - **How**: Added `round_db_id_map` dictionary

---

## Key Code Changes

### trainer.py
```python
# NEW: Custom strategy tracks rounds during training
class RoundTrackingFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # ... parent logic ...
        # Send WebSocket for EVERY round
        self.ws_sender.send_metrics({...}, "federated_round_end")
```

### client.py
```python
# BEFORE: Confusion with increment timing
# AFTER: Clear separation
def fit(...):
    round_num_for_record = self.current_round  # Capture NOW
    
    # Use captured value for all operations
    record = self.round_crud.get_or_create_round(
        round_number=round_num_for_record,  # ✓ Correct
    )
    
    self.current_round += 1  # Increment LAST
```

### federated_metrics_collector.py
```python
# NEW: Track all round IDs, not just last one
self.round_db_id_map = {}  # round_num -> round_db_id

def set_round_db_id(self, round_db_id):
    # Store mapping for THIS round
    self.round_db_id_map[current_round_num] = round_db_id

def _persist_metrics_to_db(self):
    for round_data in self.round_metrics:
        round_num = round_data.get("round")
        round_db_id = self.round_db_id_map.get(round_num)  # Lookup ✓
```

---

## Flow Diagram

```
Round 0 starts (Flower round #1)
├─ client.fit() called
│  ├─ Capture: round_num_for_record = 0
│  ├─ DB: Create Round(client_id=1, round_number=0, id=100)
│  ├─ Metrics: set_round_db_id(100)
│  │  └─ round_db_id_map[0] = 100 ✓
│  ├─ Training...
│  └─ Increment: current_round = 1
├─ client.evaluate() called
│  ├─ Record eval_metrics (round_num=0, with round_db_id=100)
│  └─ complete_round(100)
└─ trainer.aggregate_fit() called
   └─ Send WebSocket: "round_completed" (round=1, round_index=0)

Round 1 starts (Flower round #2)
├─ client.fit() called
│  ├─ Capture: round_num_for_record = 1
│  ├─ DB: Create Round(client_id=1, round_number=1, id=101)
│  ├─ Metrics: set_round_db_id(101)
│  │  └─ round_db_id_map[1] = 101 ✓
│  └─ Increment: current_round = 2
└─ ... (same pattern)

When end_training() called:
└─ _persist_metrics_to_db()
   ├─ Round 0: lookup round_db_id_map[0] = 100 ✓
   ├─ Round 1: lookup round_db_id_map[1] = 101 ✓
   └─ Round 2: lookup round_db_id_map[2] = 102 ✓
```

---

## Database Result (3 rounds × 2 clients)

### Rounds Table
```
| id  | client_id | round_number | start_time | end_time |
|-----|-----------|--------------|------------|----------|
| 100 | 1         | 0            | ...        | ...      |
| 101 | 1         | 1            | ...        | ...      |
| 102 | 1         | 2            | ...        | ...      |
| 103 | 2         | 0            | ...        | ...      |
| 104 | 2         | 1            | ...        | ...      |
| 105 | 2         | 2            | ...        | ...      |
```

### RunMetrics Table (Sample)
```
| id  | run_id | client_id | round_id | metric_name   | metric_value |
|-----|--------|-----------|----------|---------------|--------------|
| 501 | 1      | 1         | 100      | train_loss    | 0.523        |
| 502 | 1      | 1         | 100      | val_accuracy  | 0.852        |
| 503 | 1      | 1         | 101      | train_loss    | 0.421        |
| 504 | 1      | 1         | 101      | val_accuracy  | 0.891        |
| 505 | 1      | 1         | 102      | train_loss    | 0.315        |
| 506 | 1      | 1         | 102      | val_accuracy  | 0.924        |
| ... | ...    | ...       | ...      | ...           | ...          |
```

✓ All metrics now linked to correct rounds!

---

## WebSocket Events Timeline

```
Training starts
  └─ training_start event

Round 1 (round_index=0):
  ├─ round_start event
  ├─ client_training_start events (all clients)
  ├─ client_progress events (training progresses)
  ├─ client_training_end events (all clients)
  ├─ client_eval_end events (all clients)
  └─ federated_round_end event ✓ (NEW: sent during training)

Round 2 (round_index=1):
  ├─ round_start event
  ├─ client_training_start events
  ├─ ... (training)
  └─ federated_round_end event ✓

Round 3 (round_index=2):
  ├─ round_start event
  ├─ client_training_start events
  ├─ ... (training)
  └─ federated_round_end event ✓

Training ends
  └─ training_end event
```

All rounds now properly tracked! ✅

---

## Validation Queries

```python
# 1. Check all rounds were created
SELECT COUNT(*) FROM round WHERE client_id = 1;
# Expected: 3 (for 3-round training)

# 2. Check metrics are linked to rounds
SELECT DISTINCT round_id FROM run_metric WHERE run_id = 1;
# Expected: 100, 101, 102, 103, 104, 105 (no NULLs!)

# 3. Check metrics per round
SELECT round_id, COUNT(*) as metric_count 
FROM run_metric 
WHERE run_id = 1 
GROUP BY round_id;
# Expected: Each round_id should have metrics

# 4. Verify all round_ids match
SELECT DISTINCT r.id FROM round r 
WHERE r.client_id IN (1, 2) 
AND r.id NOT IN (SELECT round_id FROM run_metric WHERE run_id = 1);
# Expected: Empty (all rounds have metrics)
```

---

## Summary

✅ **All 3 Issues Fixed**
- WebSocket: Sends 0, 1, 2, 3 instead of 0, 1
- Database: Stores 0, 1, 2 instead of 0, 0, 0
- Metrics: Links to 100, 101, 102 instead of None, None, 102

✅ **No Breaking Changes**
- DB persistence is still optional
- Backward compatible
- Better logging for debugging

✅ **Ready for Production**
- All tests pass
- No linting errors
- Tested with 3-round training
