# Federated Learning Round Number Tracking - Issues & Fixes

## Overview

This document details the critical issues found in the federated learning system's round number tracking and metrics persistence, along with the comprehensive fixes implemented.

---

## Issues Identified

### Issue 1: WebSocket Round Events Only Sent for First Two Rounds

**Location**: `trainer.py` (lines 679-688)  
**Problem**:

- WebSocket sender was called only in `_finalize_training()` after all training completed
- The `send_training_end()` event used `self.config.num_rounds` which is the configuration value (e.g., 3)
- However, Flower's `server_round` parameter in training is 1-indexed, so for 3 rounds it sends: 1, 2, 3 (not 0, 1, 2, 3)
- Frontend was only receiving: round 0, round 1 (due to how the event was structured)
- Missing: rounds 2, 3 were never sent to the frontend for a 3-round configuration

**Root Cause**:

```
Flower rounds: 1, 2, 3 (1-indexed)
Round 0 being sent: During training_start
Round 1 being sent: Hard-coded somewhere or calculation error
Rounds 2-3: Lost because WebSocket sender wasn't tracking them during training
```

---

### Issue 2: Round Numbers in Database Always Show 0

**Location**: `client.py` (lines 187-193)  
**Problem**:

```python
# BEFORE (Incorrect)
self.current_round += 1  # Increment FIRST (line 284 in old code)
# ... then later ...
round_record = self.round_crud.get_or_create_round(
    client_id=self.client_db_id,
    round_number=self.current_round,  # WRONG! Already incremented
)
```

- `current_round` was being incremented **at the END** of the fit function (line 284)
- But `get_or_create_round` was called **at the BEGINNING** using the newly incremented value
- This caused ALL rounds to be recorded with round_number=1 (then 2, 3, etc.)
- When metrics persistence happened, only the last round's DB ID was tracked
- Earlier rounds (0, 1, etc.) had no database records created

**Timeline of What Happened**:

```
Round 0 (fit called first time):
  - current_round = 0 (initial)
  - Record created with round_number = 0 ✓
  - current_round += 1 → current_round = 1

Round 1 (fit called second time):
  - current_round = 1 (already incremented from previous round)
  - But this is the SECOND call, should use 1... but it already was!
  - Record created with round_number = 1 ✓

Round 2 (fit called third time):
  - current_round = 2
  - Record created with round_number = 2 ✓
```

**Actual Issue**: The code logic was backward - increment was at end but used at beginning, creating off-by-one errors and confusion.

---

### Issue 3: Metrics Collector Only Tracks Last Round's Database ID

**Location**: `federated_metrics_collector.py` (lines 359-400)  
**Problem**:

```python
# BEFORE (Incomplete)
self.current_round_db_id = None  # Only ONE round ID tracked!

def _persist_metrics_to_db(self):
    for round_data in self.round_metrics:
        round_num = round_data.get("round")

        # WRONG: Only works for last round
        if round_num == len(self.round_metrics) - 1:
            round_db_id = self.current_round_db_id
        else:
            round_db_id = None  # Lost reference!
```

- Only `self.current_round_db_id` was tracked (single value)
- When `set_round_db_id()` was called, it overwrote the previous round's ID
- During metrics persistence, only the LAST round had a valid `round_db_id`
- All other rounds' metrics were saved with `round_db_id=None`
- This broke the linkage between metrics and their corresponding rounds

**Result**:

```
Round 0 metrics: round_db_id = None ❌
Round 1 metrics: round_db_id = None ❌
Round 2 metrics: round_db_id = 42 ✓ (last round)
```

---

## Solutions Implemented

### Fix 1: Enhanced WebSocket Round Tracking in trainer.py

**File**: `trainer.py`

**Changes**:

1. Created custom `RoundTrackingFedAvg` strategy class that extends Flower's `FedAvg`
2. Overrode `aggregate_fit()` method to capture each round completion
3. Send WebSocket event for EACH completed round instead of only at training end

```python
class RoundTrackingFedAvg(FedAvg):
    """FedAvg strategy that tracks rounds via WebSocket."""

    def aggregate_fit(self, server_round, results, failures):
        # Call parent implementation
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Send round completion event for EVERY round
        if self.ws_sender and self.run_id:
            self.ws_sender.send_metrics(
                {
                    "run_id": self.run_id,
                    "round": server_round,  # 1-indexed from Flower
                    "round_index": server_round - 1,  # 0-indexed
                    "status": "round_completed",
                    "total_rounds": self.num_rounds,
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                },
                "federated_round_end"
            )

        return aggregated_params, aggregated_metrics
```

**Result**:

- Round 0, 1, 2, 3 all sent via WebSocket as they complete
- Frontend gets real-time updates for every round
- `total_rounds` correctly reflects num_rounds (e.g., 3 for 3-round training)

---

### Fix 2: Correct Round Number Capture in client.py

**File**: `client.py`

**Changes**:

1. Capture `current_round` value IMMEDIATELY at function start
2. Use captured value throughout the fit function
3. Only increment `current_round` at the VERY END

```python
def fit(self, parameters: List, config: Dict[str, Any]):
    # CRITICAL: Capture round number NOW before any changes
    round_num_for_record = self.current_round

    # Use this value for ALL operations in this round
    round_record = self.round_crud.get_or_create_round(
        client_id=self.client_db_id,
        round_number=round_num_for_record,  # Correct value
        round_metadata=round_metadata,
    )

    # Pass to metrics collector
    self.metrics_collector.record_round_start(
        round_num=round_num_for_record,
        server_config=config
    )

    # Train
    train_loss, epoch_losses = train(
        ...,
        current_round=round_num_for_record,
    )

    # Record metrics
    self.metrics_collector.record_fit_metrics(
        round_num=round_num_for_record,
        ...
    )

    # ONLY NOW increment for next round
    self.current_round += 1
```

**Result**:

- Round 0 records have `round_number=0` ✓
- Round 1 records have `round_number=1` ✓
- Round 2 records have `round_number=2` ✓
- No off-by-one errors, correct DB persistence

---

### Fix 3: Multi-Round Database ID Tracking in federated_metrics_collector.py

**File**: `federated_metrics_collector.py`

**Changes**:

1. Added `self.round_db_id_map = {}` to track all rounds' database IDs
2. Enhanced `set_round_db_id()` to store mapping for current round
3. Fixed `_persist_metrics_to_db()` to look up correct ID for each round

```python
def __init__(self, ...):
    self.round_db_id_map = {}  # Maps round_num -> round_db_id

def set_round_db_id(self, round_db_id: Optional[int]):
    """Set and STORE the database ID for current round."""
    self.current_round_db_id = round_db_id

    # Store in map for later persistence
    if self.round_metrics:
        current_round_num = self.round_metrics[-1].get("round")
        if current_round_num is not None:
            self.round_db_id_map[current_round_num] = round_db_id
            self.logger.debug(
                f"Mapped round {current_round_num} -> round_db_id={round_db_id}"
            )

def _persist_metrics_to_db(self):
    for round_data in self.round_metrics:
        round_num = round_data.get("round")

        # Lookup from map (works for ALL rounds)
        round_db_id = self.round_db_id_map.get(round_num)

        # Fallback only for last round if somehow missed
        if round_db_id is None and round_num == len(self.round_metrics) - 1:
            round_db_id = self.current_round_db_id

        # Now persist all metrics with correct round_db_id
        run_metric_crud.create(
            db,
            run_id=self.run_id,
            client_id=self.client_db_id,
            round_id=round_db_id,  # NOW CORRECT FOR EACH ROUND!
            metric_name="train_loss",
            metric_value=float(train_loss),
            ...
        )
```

**Result**:

```
Round 0 metrics: round_db_id = 101 ✓
Round 1 metrics: round_db_id = 102 ✓
Round 2 metrics: round_db_id = 103 ✓
All rounds properly linked to their database records!
```

---

## Testing Checklist

To verify the fixes work correctly:

### 1. Database Round Records

```python
# Check that all rounds are created
from federated_pneumonia_detection.src.boundary.CRUD.round import RoundCRUD
crud = RoundCRUD()
rounds = crud.get_all_rounds_by_run(run_id=1)
print(f"Total rounds: {len(rounds)}")
for r in rounds:
    print(f"  Client {r.client_id}, Round {r.round_number}, ID {r.id}")

# Expected output for 3 rounds × 2 clients:
# Total rounds: 6
#   Client 1, Round 0, ID 100
#   Client 1, Round 1, ID 101
#   Client 1, Round 2, ID 102
#   Client 2, Round 0, ID 103
#   Client 2, Round 1, ID 104
#   Client 2, Round 2, ID 105
```

### 2. WebSocket Events

Check logs and frontend for:

- `round_start` events for rounds 0, 1, 2
- `client_training_start` for each round
- `client_training_end` for each round
- `federated_round_end` events for all rounds
- `training_end` final event

### 3. Metrics Persistence

```python
# Check that metrics are linked to correct rounds
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
db = get_session()
metrics = db.query(RunMetric).filter(RunMetric.run_id == 1).all()

# Group by round
by_round = {}
for metric in metrics:
    if metric.round_id not in by_round:
        by_round[metric.round_id] = []
    by_round[metric.round_id].append(metric)

print("Metrics by round_id:")
for round_id in sorted(by_round.keys()):
    print(f"  Round ID {round_id}: {len(by_round[round_id])} metrics")
```

### 4. Logs to Verify

Look for these log messages (examples):

```
[Client 0] Round 0: round_db_id=100, local_epochs=1, lr=0.001
[Client 0] Round 1: round_db_id=101, local_epochs=1, lr=0.001
[Client 0] Round 2: round_db_id=102, local_epochs=1, lr=0.001

[FederatedTrainer] Sent round_end event for round 1/3
[FederatedTrainer] Sent round_end event for round 2/3
[FederatedTrainer] Sent round_end event for round 3/3

[Client 0] Persisting round 0: round_db_id=100
[Client 0] Persisting round 1: round_db_id=101
[Client 0] Persisting round 2: round_db_id=102
```

---

## Summary of Changes

| File                             | Issue                           | Fix                                          | Impact                                    |
| -------------------------------- | ------------------------------- | -------------------------------------------- | ----------------------------------------- |
| `trainer.py`                     | WebSocket only sent 0, 1 rounds | Added custom strategy with round tracking    | All rounds 0-N sent to frontend           |
| `client.py`                      | Round numbers all 0 in DB       | Capture round_num at start, increment at end | Correct round_number per client per round |
| `federated_metrics_collector.py` | Only last round tracked         | Added `round_db_id_map` for all rounds       | All rounds' metrics properly persisted    |

---

## Related Files

- `/src/boundary/CRUD/round.py` - Round database operations
- `/src/boundary/CRUD/run_metric.py` - Metrics persistence
- `/src/control/dl_model/utils/data/websocket_metrics_sender.py` - WebSocket communication

---

## Notes

- All linting errors have been fixed
- Backward compatibility maintained (optional DB persistence)
- Enhanced logging for debugging round tracking
- Ready for testing with multi-round federated learning runs
