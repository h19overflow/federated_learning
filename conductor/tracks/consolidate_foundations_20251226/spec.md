# Track Specification: Consolidate Project Foundations

## Objective
The primary objective of this track is to consolidate the existing Brownfield project by auditing the codebase for architectural compliance, verifying and improving test coverage to meet the >80% standard, and establishing a rigorous performance baseline for comparing Centralized and Federated Learning approaches.

## Context
The project is a Federated Pneumonia Detection System with an existing codebase comprising a FastAPI backend, PyTorch Lightning models, Flower-based federated learning, and a React frontend. The project needs to be brought into full alignment with the new "Conductor" setup, ensuring high code quality and readiness for the Final Year Project (FYP) evaluation.

## Requirements

### 1. Architecture & Code Audit
- **Goal:** Ensure all components adhere to the Entity-Control-Boundary (ECB) pattern.
- **Actions:**
    - Scan `src/entities`, `src/control`, `src/boundary` for strictly separated concerns.
    - Identify and document any circular dependencies or logic leakage (e.g., business logic in boundary layers).
    - Verify strict type hinting (Python) and TypeScript interfaces for all public APIs.

### 2. Testing & Coverage
- **Goal:** Achieve >80% code coverage across the backend and frontend.
- **Actions:**
    - configured `pytest` and `jest`/`vitest` to generate coverage reports.
    - Audit existing tests for quality (mocking, edge cases).
    - Write missing unit and integration tests to close coverage gaps.

### 3. Performance Baselining
- **Goal:** Establish a reproducible benchmark for Centralized vs. Federated Learning.
- **Actions:**
    - Create a standardized "Check" script that runs a small-scale training run for both modes.
    - Record initial metrics (Accuracy, Loss, Time-to-convergence) to serve as a baseline.
    - Ensure the WebSocket metrics pipeline is functioning correctly for real-time visualization.

## Acceptance Criteria
- [ ] Full Architectural Audit Report generated.
- [ ] Code Coverage Report showing >80% coverage.
- [ ] Successful execution of both Centralized and Federated training flows on a sample dataset.
- [ ] Baseline Performance Report documented.
