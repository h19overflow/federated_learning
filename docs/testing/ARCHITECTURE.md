# Testing Architecture

**Scope**: `tests/` directory covering Unit, Integration, and API testing layers.
**Target**: >90% code coverage.

---

## Test Pyramid & Strategy

```mermaid
graph TD
    API["API Tests (Integration)<br/>tests/api/"] --> INT["Integration Tests<br/>tests/integration/"]
    INT --> UNIT["Unit Tests<br/>tests/unit/"]
    
    style API fill:#e1f5ff,stroke:#01579b
    style INT fill:#fff3e0,stroke:#e65100
    style UNIT fill:#f3e5f5,stroke:#4a148c
```

## Directory Structure & Mapping

| Source Path | Test Path | Description |
|---|---|---|
| `src/entities/` | `tests/unit/entities/` | Entity class validation |
| `src/control/` | `tests/unit/control/` | Core logic and algorithms |
| `src/boundary/` | `tests/unit/boundary/` | DB models and interfaces |
| `src/api/` | `tests/api/` | REST/WebSocket endpoints |
| *System Flows* | `tests/integration/` | Cross-component workflows |

## Core Configuration Components

| File | Purpose | Key Settings |
|---|---|---|
| `tests/pytest.ini` | Test runner config | Markers, coverage settings, log levels |
| `tests/conftest.py` | Shared fixtures | `db_session`, `mock_inference_service`, `temp_data` |
| `tests/test_config.yaml` | Test constants | `batch_size: 8`, `epochs: 2` (Speed optimization) |
| `tests/run_fl_tests.py` | FL Runner | Custom runner for federated learning suites |

---

## Fixture Architecture

Global fixtures are defined in `tests/conftest.py` and modularized in `tests/fixtures/`.

```mermaid
classDiagram
    class Conftest {
        +sample_config
        +temp_data_structure
        +db_session
        +client
    }
    class SampleDataFactory {
        +create_dummy_image()
        +create_sample_metadata()
    }
    class MockComponents {
        +create_mock_data_processor()
        +create_mock_trainer()
    }
    
    Conftest ..> SampleDataFactory : Uses
    Conftest ..> MockComponents : Uses
    TestClass ..> Conftest : Injects
```

## File Reference

| Component | File | Key Lines |
|---|---|---|
| Global Setup | `tests/conftest.py` | 15-80 |
| Data Factory | `tests/fixtures/sample_data.py` | 12-100 |
| Runner | `tests/run_fl_tests.py` | 10-60 |
