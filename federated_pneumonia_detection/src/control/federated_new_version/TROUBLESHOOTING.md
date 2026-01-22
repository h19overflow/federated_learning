#+#+#+#+------------------------------------------------------------------
TROUBLESHOOTING
Federated (Flower Message API) - Common Failure Modes
#+#+#+#+------------------------------------------------------------------

## Global Model Not Updating Across Rounds (FedAvg)

### Symptom
- Server-side evaluation metrics stay identical across rounds (loss/acc/auroc unchanged)
- `train_metrics_clientapp` ends up empty or missing on the server
- Clients appear to train (local parameter changes logged), but the global model does not improve

### What Actually Happened
Flower FedAvg (Message API, `flwr[simulation]>=1.22.0`) can *skip aggregation* when client replies
do not satisfy strict consistency requirements.

When aggregation is skipped:
- the server keeps using the previous global weights
- server-side evaluation runs, but evaluates the same model each round

### Root Cause
FedAvg validates reply consistency before averaging ArrayRecords/MetricRecords. Aggregation is
skipped if either of these is true:

- The weight key is missing: `weighted_by_key` defaults to `num-examples` (hyphen).
- The weight value is not a scalar numeric type (enforce `int` for safety).
- Client MetricRecords do not have *identical key sets* (one client missing `val_auroc`, another
  including it, etc.).

In our case, the per-client FIT metrics were derived from training history in a way that could
produce inconsistent keys across clients/rounds, which caused Flower's validation to fail and
drop aggregation.

### The Oversight (Lessons)
This was an interface/abstraction-layer issue:

- We did not fully verify which Strategy hook Flower calls in the Message API.
  - Training aggregation is performed via `aggregate_train` (ArrayRecord + MetricRecord), not a
    custom `aggregate_fit` hook.
- We did not enforce the framework invariants (fixed MetricRecord schema + correct weight key).

Net effect: client training succeeded, but the server never applied updates.

### Fix (What Made It Work)

1) Enforce a fixed, deterministic FIT metric schema
- Extract metrics deterministically (use last epoch only)
- Always emit the same metric keys from every client, every round
- Use numeric defaults for missing metrics (e.g. `0.0`) instead of omitting keys
- Ensure `epoch` is `int` and other metrics are `float`

2) Ensure weighted aggregation key is correct and well-typed
- Include `num-examples` (hyphen) in the MetricRecord
- Enforce it as `int`
- Pass `weighted_by_key="num-examples"` into FedAvg init (defensive, avoids version surprises)

3) Add visibility at the correct layer
- Override `aggregate_train` to log:
  - each client's metric keys
  - `num-examples` presence/type
  - aggregated metric keys
- If aggregation fails, print per-client metrics to make key mismatches obvious

### How To Verify The Fix
- Server logs show, each round:
  - `Aggregating TRAIN from N clients`
  - each client has the same `metric_keys` and `num-examples` is int
  - `TRAIN aggregated metric_keys=...`
- Server evaluation metrics change across rounds (not byte-identical)

### Quick Checklist For Next Time (Framework Integrations)
- Confirm exact framework version and read the API docs/source for that version
- Identify which hooks are actually invoked in the workflow (train vs eval aggregation)
- Write down the reply schema invariants (required keys + types)
- Add logging at the framework boundary first (before adding features)
- Run a 2-client / 2-round smoke test and confirm global weights/metrics change
