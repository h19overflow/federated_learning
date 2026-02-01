# Run Metrics Sequence Diagram

**API**: `GET /api/runs/{runId}/metrics`  
**Component**: `ResultsVisualization.tsx` (lines 30-39)

---

## Step 1: Component to API Request

**Files**: 
- `ResultsVisualization.tsx` (lines 30-39)
- `useResultsVisualization.ts` (line 45)
- `api.ts` (lines 120-125)

```mermaid
sequenceDiagram
    participant C as ResultsVisualization.tsx
    participant H as useResultsVisualization.ts
    participant A as api.ts

    C->>H: useResultsVisualization({config, runId})
    Note right of H: Hook receives runId
    
    H->>A: getRunMetrics(runId)
    Note right of A: api.ts line 120-125
    
    A->>E: GET /api/runs/:runId/metrics
    Note right of E: HTTP request to backend
```

**Key Code**:
```typescript
// ResultsVisualization.tsx line 30-39
const {
  activeResults,           // ← From getRunMetrics
  trainingHistoryData,     // ← Transformed
  confusionMatrix,         // ← Transformed
  metricsChartData,        // ← Transformed
} = useResultsVisualization({ config, runId });
```

---

## Step 2: Backend Processing

**Files**:
- `runs_analytics.py` (lines 40-50)
- `facade.py` (lines 60-70)
- `metrics_service.py` (lines 80-110)

```mermaid
sequenceDiagram
    participant E as runs_analytics.py
    participant F as AnalyticsFacade
    participant S as MetricsService
    participant CR as RunCRUD

    E->>F: analytics.metrics.get_run_metrics(db, runId)
    Note right of F: facade.py line 60-70
    
    F->>S: metrics_service.get_run_metrics(db, runId)
    Note right of S: metrics_service.py line 80-110
    
    S->>S: Check cache
    
    alt Cache Miss
        S->>CR: run_crud.get_with_metrics(db, runId)
        Note right of CR: CRUD layer
    end
```

**Key Code**:
```python
# metrics_service.py line 80-110
def get_run_metrics(self, db: Session, run_id: int):
    key = cache_key("get_run_metrics", (run_id,), {})
    
    def _compute():
        run = self._run_crud.get_with_metrics(db, run_id)
        persisted_stats = self._get_persisted_stats(db, run)
        return transform_run_to_results(run, persisted_stats)
    
    return self._cache.get_or_set(key, _compute)
```

---

## Step 3: Database Queries

**File**: `run.py` (lines 45-60)

```mermaid
sequenceDiagram
    participant CR as RunCRUD
    participant D as Database

    CR->>D: SELECT * FROM runs WHERE id = ?
    Note right of D: runs table
    D-->>CR: Run record
    
    CR->>D: SELECT * FROM run_metrics WHERE run_id = ?
    Note right of D: run_metrics table
    D-->>CR: Metrics list
    
    CR->>D: SELECT * FROM server_evaluations WHERE run_id = ?
    Note right of D: server_evaluations table (federated)
    D-->>CR: Evaluations list
```

**Queries**:

| Table | Query | Purpose |
|-------|-------|---------|
| `runs` | `SELECT * FROM runs WHERE id = ?` | Get experiment metadata |
| `run_metrics` | `SELECT * FROM run_metrics WHERE run_id = ?` | Per-epoch metrics (centralized) |
| `server_evaluations` | `SELECT * FROM server_evaluations WHERE run_id = ? ORDER BY round_number` | Per-round metrics (federated) |

---

## Step 4: Transform & Response

**File**: `transformers.py` (lines 50-120)

```mermaid
sequenceDiagram
    participant S as MetricsService
    participant T as transformers.py
    participant H as useResultsVisualization.ts
    participant C as ResultsVisualization.tsx

    S->>T: transform_run_to_results(run, persisted_stats)
    Note right of T: transformers.py line 50-120
    
    T->>T: Check run.training_mode
    
    alt Federated
        T->>T: Read from run.server_evaluations
    else Centralized
        T->>T: Read from run.metrics
    end
    
    T->>T: Build training_history
    T->>T: Extract final_metrics
    T->>T: Build confusion_matrix
    
    T-->>S: ExperimentResults dict
    S-->>H: Return results
    
    H->>H: Transform props
    Note right of H: trainingHistoryData<br/>confusionMatrix<br/>metricsChartData
    
    H-->>C: activeResults + transformed
```

**Transformations**:

| Source | Transformed To | Component Prop |
|--------|----------------|----------------|
| `training_history` | Chart format | `trainingHistoryData` |
| `confusion_matrix` | 2D array | `confusionMatrix` |
| `metadata` | Bar chart data | `metricsChartData` |

---

## File Reference

| Layer | File | Key Lines |
|-------|------|-----------|
| Component | `ResultsVisualization.tsx` | 30-39 |
| Hook | `useResultsVisualization.ts` | 45, 85-120 |
| API Service | `api.ts` | 120-125 |
| API Endpoint | `runs_analytics.py` | 40-50 |
| Analytics | `facade.py` | 60-70 |
| Service | `metrics_service.py` | 80-110 |
| Transformer | `transformers.py` | 50-120 |
| CRUD | `run.py` | 45-60 |
