# Federated Learning Test Flow

**Process**: Simulation of Server-Client coordination for Federated Learning.
**Entry Point**: `tests/run_fl_tests.py` or `pytest -m federated`

---

## Step 1: Simulation Setup

**Action**: The test runner initializes the mocked environment for federated learning, creating virtual client datasets.

```mermaid
sequenceDiagram
    participant Test as TestRunner
    participant Mock as MockDatasets
    participant Config as TestConfig
    
    Test->>Mock: federated_datasets()
    Mock->>Mock: Generate N partitions
    Mock->>Test: List[DataFrame]
    Test->>Config: Set num_clients=N
```

**Key Code**:
```python
# tests/fixtures/sample_data.py lines 180-210
@staticmethod
def federated_datasets() -> List[pd.DataFrame]:
    # Creates multiple datasets with different distributions
    # Simulates heterogeneous client data (non-IID)
    pass
```

---

## Step 2: Server-Client Round Execution

**Action**: The test simulates a training round where the server aggregates weights from clients.

```mermaid
sequenceDiagram
    participant Server as FL Server
    participant Client1 as FlowerClient 1
    participant Client2 as FlowerClient 2
    participant Aggr as Aggregator
    
    Server->>Client1: fit(parameters)
    Client1->>Client1: Local Train (Mocked/Fast)
    Client1->>Server: Return new_parameters
    
    Server->>Client2: fit(parameters)
    Client2->>Client2: Local Train (Mocked/Fast)
    Client2->>Server: Return new_parameters
    
    Server->>Aggr: aggregate_fit(results)
    Aggr->>Server: Updated Global Model
```

**Key Code**:
```python
# tests/integration/federated_learning/test_fl_flow.py (Conceptual)
def test_round_aggregation(mock_server, mock_clients):
    # Simulate round
    results = [c.fit(params) for c in mock_clients]
    aggregated = mock_server.aggregate(results)
    assert aggregated is not None
```

---

## File Reference

| Layer | File | Description |
|-------|------|-------------|
| Runner | `tests/run_fl_tests.py` | CLI entry point for FL tests |
| Data | `tests/fixtures/sample_data.py` | `federated_datasets` factory |
| Config | `tests/test_config.yaml` | `federated` section params |
