# Federated Confusion Matrix Persistence - Athena Plan

## Summary
Persist both (1) server-side confusion matrices computed on the server test set and (2) client-aggregated confusion matrices produced by the Flower strategy so the API can reliably return a confusion matrix after training and the system can later compare “server-evaluated” vs “client-reported/aggregated” behavior.

## Context Gathered

### Patterns Found
- Federated API metrics source: `federated_pneumonia_detection/src/api/endpoints/runs_endpoints/shared/utils.py` reads federated runs from `Run.server_evaluations` (not `Run.metrics`).
- Confusion matrix response format: API builds `confusion_matrix` from `val_cm_tp/tn/fp/fn` in the *last epoch/round* and returns nested keys `true_positives/true_negatives/false_positives/false_negatives` plus derived stats.
- Federated server evaluation emits CM keys: `federated_pneumonia_detection/src/control/federated_new_version/core/server_evaluation.py` returns `server_cm_tp/tn/fp/fn`.
- ServerEvaluation persistence is centralized at end-of-run: `federated_pneumonia_detection/src/control/federated_new_version/core/server_app.py` calls `_persist_server_evaluations()`.
- ServerEvaluation schema already supports CM: `federated_pneumonia_detection/src/boundary/models/server_evaluation.py` has `true_positives/true_negatives/false_positives/false_negatives` integer columns.
- Strategy aggregates client-reported CM: `federated_pneumonia_detection/src/control/federated_new_version/core/custom_strategy.py` extracts `cm_tp/tn/fp/fn` and broadcasts via `send_round_metrics()`, but does not persist.

### Technology Stack
- Backend: FastAPI + SQLAlchemy models (`Run`, `RunMetric`, `ServerEvaluation`) with Alembic baseline migration.
- Federated: Flower strategy (`FedAvg`) + custom server evaluation.

### Similar Implementations
- Centralized persistence uses `RunMetric` (per-epoch, metric_name/value): `federated_pneumonia_detection/src/boundary/models/run_metric.py` + `federated_pneumonia_detection/src/boundary/CRUD/run.py:persist_metrics()`.
- Federated transformation already maps `ServerEvaluation.true_*` to `val_cm_*` in API shaping: `federated_pneumonia_detection/src/api/endpoints/runs_endpoints/shared/utils.py`.

### Knowledge Gaps
- Whether frontend needs CM live from WebSocket (TypeScript `RoundMetricsData` currently does not type CM keys). This does not block DB/API persistence.

## Approach

### Goal Alignment
- Make `/api/runs/{run_id}/metrics` return a non-null `confusion_matrix` for federated runs by ensuring `ServerEvaluation.true_*` columns are populated.
- Persist client-aggregated CM as a separate “aggregated” record set for audit/comparison, without breaking existing federated API behavior.

### Architecture Decisions

#### Question 1: Where to store client-aggregated confusion matrix?

Option A: Store in an “Evaluation” table (as described in task)
- Reality check: there is no `Evaluation` model in this repo; centralized metrics are stored in `run_metrics` (`RunMetric`).

Option B: Store in `server_evaluations`
- Interpretation: treat `ServerEvaluation` as “global metrics per round” and store either server-evaluated or client-aggregated values.
- Problem: we likely want BOTH, and overloading the same columns creates ambiguity (which CM is shown to the user?).

Option C: Store in `run_metrics` (`RunMetric`) with `context="aggregated"` (recommended)
- Works with current schema (no migration).
- Keeps concerns separated: `ServerEvaluation` = server-side centralized evaluation; `RunMetric(context=aggregated)` = strategy-aggregated client-reported metrics.
- Enables future API expansion to show both series, while keeping current federated response stable.

Selected: Option C (store client-aggregated CM in `run_metrics` with `context="aggregated"`).

#### Question 2: When to persist client-aggregated metrics?

Option A: After each round (inside `aggregate_evaluate()`)
- Pros: durable even if process crashes; supports mid-training queries; matches WebSocket cadence.
- Cons: more DB writes (small in practice: rounds are low, metrics are few).

Option B: Only final round
- Pros: minimal writes.
- Cons: loses per-round history; still vulnerable to crash before final persistence.

Option C: Configurable (default per-round)
- Pros: flexibility for dev vs production.
- Cons: slightly more config surface.

Selected: Option C with default = Option A (persist per round; allow disabling via config/env).

#### Question 3: Key naming convention?

Observed keys today
- Client-side metrics (reported): `test_cm_tp/tn/fp/fn` (and potentially `val_cm_*`).
- Strategy broadcast keys: `cm_tp/tn/fp/fn`.
- Server evaluation keys: `server_cm_tp/tn/fp/fn`.
- API training-history keys: expects `val_cm_tp/tn/fp/fn` to build `confusion_matrix`.

Recommendation (unified convention by layer)
- DB (authoritative storage):
  - `ServerEvaluation`: `true_positives/true_negatives/false_positives/false_negatives` columns.
  - `RunMetric`: store confusion matrix as `metric_name` values `val_cm_tp/val_cm_tn/val_cm_fp/val_cm_fn` with `context="aggregated"` and `step=<round_number>`.
- Transport (Flower/server eval): keep `server_cm_*` as-is (already used and tested), but ensure persistence extracts these.
- Strategy broadcast (WebSocket): keep existing `cm_*` to avoid breaking current consumers; optionally include aliases (`val_cm_*`) in the payload later if frontend wants strict typing.
- API response (frontend contract): keep the current nested `confusion_matrix` object with `true_*` keys; it is already implemented.

## Trade-offs
- Option A (store aggregated CM in `server_evaluations`): simplest for API reuse, but conflates two semantics (server-evaluated vs client-aggregated) and makes comparison impossible without extra fields.
- Option B (new `ClientAggregatedMetrics` table): clean separation and strong typing, but requires schema migration + CRUD + API shaping work.
- Option C (store aggregated CM in `run_metrics` with `context`): no migration, keeps server-evaluations clean, supports future comparison; requires minimal new persistence code and (optionally) future API enhancements.
- Selected: Option C, because it preserves the current API contract while adding durable storage for strategy aggregates.

## Execution Steps

1. Fix server-side CM extraction (Gap 2)
   - Update `federated_pneumonia_detection/src/control/federated_new_version/core/utils.py` in `_persist_server_evaluations()` to include:
     - `server_cm_tp`, `server_cm_tn`, `server_cm_fp`, `server_cm_fn` in `extracted_metrics`.
   - Rationale: `ServerEvaluationCRUD.create_evaluation()` already knows how to map `server_cm_*` into `true_*` DB columns; it just never receives them.

2. Add persistence for client-aggregated metrics (Gap 1)
   - Update `federated_pneumonia_detection/src/control/federated_new_version/core/custom_strategy.py`:
     - After computing `round_metrics` in `aggregate_evaluate()`, persist them using `RunMetric` entries with:
       - `run_id=self.run_id`
       - `step=server_round`
       - `context="aggregated"`
       - `metric_name` using the API-compatible names:
         - `val_loss`, `val_accuracy`, `val_precision`, `val_recall`, `val_f1`, `val_auroc`
         - `val_cm_tp`, `val_cm_tn`, `val_cm_fp`, `val_cm_fn`
     - DB wiring: reuse `run_metric_crud.bulk_create()` (or a small helper) and commit per round.
   - Note: Keep the WebSocket broadcast unchanged for now.

3. Decide whether federated API should surface client-aggregated series (optional, future-facing)
   - Leave `/api/runs/{run_id}/metrics` behavior unchanged (still uses `server_evaluations` for federated).
   - If comparison is desired in the same response, add a new field (e.g., `aggregated_training_history`) or a new endpoint (e.g., `/api/runs/{run_id}/aggregated-metrics`).
   - Do not overload existing `training_history` unless the frontend is explicitly updated.

4. Backward compatibility checks
   - Ensure centralized pipeline continues to read `RunMetric` without filtering on `context`.
   - Ensure federated pipeline continues to read `ServerEvaluation` for primary metrics.

5. Tests
   - Extend/adjust existing tests to cover the fixed extraction path:
     - Add a unit test for `_persist_server_evaluations()` verifying it passes `server_cm_*` into `server_evaluation_crud.create_evaluation()`.
     - Run existing integration tests:
       - `tests/integration/test_confusion_matrix_federated.py`
       - `tests/integration/test_confusion_matrix_end_to_end.py`
   - Add a unit test for strategy persistence (mock DB session + assert `run_metric_crud.bulk_create` called with `val_cm_*` and `context="aggregated"`).

## Risk Assessment
- Race conditions / concurrent writes: Flower strategy aggregation is effectively single-threaded per round; still mitigate by using short-lived sessions and committing per round.
- Key mismatches: mitigate by defining a single mapping function for `round_metrics -> RunMetric metric_name` and by keeping transport keys (`server_cm_*`, `cm_*`) unchanged.
- Transaction failures: mitigate with try/except, rollback, and logging; ensure WebSocket sending is not blocked by DB errors (fail-open for UI, fail-closed for persistence).
- Impact on centralized training: minimal if `RunMetric` additions are isolated to federated runs and use `context="aggregated"`.

## Validation Criteria
- [ ] Federated run produces non-null `ServerEvaluation.true_*` values in DB for each round.
- [ ] `/api/runs/{run_id}/metrics` returns `confusion_matrix` with `true_positives/true_negatives/false_positives/false_negatives` for federated runs.
- [ ] Frontend displays the confusion matrix for a completed federated run using the API response.
- [ ] Centralized mode still returns confusion matrix from `RunMetric(val_cm_*)` unchanged.
- [ ] Strategy-aggregated confusion matrix is queryable in DB (as `RunMetric(context="aggregated")`) for audit/comparison.

## Dependencies
- Existing DB schema must match SQLAlchemy models (baseline Alembic migration assumes tables exist).
- A valid `run_id` must be passed into `ConfigurableFedAvg(..., run_id=...)` for strategy-side persistence.
- Database connectivity must be available during training (same requirement as current `_persist_server_evaluations()` path).
