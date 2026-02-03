# Integration Test Flows

**Process**: Verification of cross-component interactions (Service-to-Service, Training-to-Inference).
**Entry Point**: `tests/integration/`

---

## Flow 1: Analytics Pipeline Integration

**Goal**: Verify that the Analytics Facade correctly orchestrates Metrics and Summary services using a real (in-memory) database.

```mermaid
sequenceDiagram
    participant Test as Test Case
    participant Facade as AnalyticsFacade
    participant Service as MetricsService
    participant DB as SQLite DB
    
    Note right of Test: Setup: Seed DB with Run & Metrics
    Test->>DB: Seed Data
    
    Test->>Facade: get_run_summary(run_id)
    Facade->>Service: get_metrics(run_id)
    Service->>DB: Query Aggregates
    DB-->>Service: Return Rows
    Service-->>Facade: Formatted Metrics
    
    Facade-->>Test: Complete Summary Object
    Test->>Test: Assert Data Integrity
```

**Key Code**:
```python
# tests/integration/control/analytics/test_analytics_integration.py
def test_full_analytics_flow_centralized():
    # ... Arrange (Seed DB) ...
    facade = AnalyticsFacade(summary=summary_service, metrics=metrics_service)
    result = facade.get_run_summary(session, run.id)
    assert result.metrics["best_accuracy"] == 0.9
```

---

## Flow 2: Model Training -> Inference Handover

**Goal**: Verify that a model checkpoint saved by the Training module can be successfully loaded and used by the Inference Engine.

```mermaid
sequenceDiagram
    participant Test as Test Case
    participant Train as LitResNetEnhanced
    participant FS as File System
    participant Infer as InferenceEngine
    
    Note right of Test: Phase 1: Training Output
    Test->>Train: Initialize Model
    Test->>FS: torch.save(checkpoint)
    
    Note right of Test: Phase 2: Inference Input
    Test->>Infer: __init__(checkpoint_path)
    Infer->>FS: load_from_checkpoint()
    Infer->>Infer: freeze_backbone()
    
    Test->>Infer: predict(dummy_image)
    Infer-->>Test: Prediction Result
```

**Key Code**:
```python
# tests/integration/control/model_inference/test_model_loading_integration.py
def test_model_loading_integration(tmp_path):
    # Save dummy checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Load into Engine
    engine = InferenceEngine(checkpoint_path=checkpoint_path)
    result = engine.predict(dummy_img)
    assert result[0] in ["PNEUMONIA", "NORMAL"]
```

---

## File Reference

| Layer | File | Description |
|-------|------|-------------|
| Analytics Test | `tests/integration/control/analytics/test_analytics_integration.py` | Service composition test |
| Model Test | `tests/integration/control/model_inference/test_model_loading_integration.py` | Serialization/Deserialization test |
