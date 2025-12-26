# Track Plan: Consolidate Project Foundations

## Phase 1: Architecture & Code Audit
*Objective: Ensure the codebase adheres to the ECB pattern and strict typing standards.*

- [ ] Task: Audit `src/entities` for pure data structures and business rules
    - **Step:** Review all files in `federated_pneumonia_detection/src/entities`.
    - **Step:** Ensure no database or API dependencies exist in entities.
    - **Step:** Verify Pydantic models and Python dataclasses are used correctly.

- [ ] Task: Audit `src/boundary` for external interface isolation
    - **Step:** Review `federated_pneumonia_detection/src/boundary`.
    - **Step:** Ensure all database and file I/O operations are encapsulated here.
    - **Step:** Verify that boundary layers do not contain complex business logic.

- [ ] Task: Audit `src/control` for orchestration logic
    - **Step:** Review `federated_pneumonia_detection/src/control`.
    - **Step:** Confirm that controllers orchestrate the flow between entities and boundaries.

- [ ] Task: Enforce Type Hinting and Linting
    - **Step:** Run `mypy` (or equivalent) to identify missing type hints.
    - **Step:** Run `ruff` or `pylint` to identify style violations.
    - **Step:** Fix critical typing and linting errors.

- [ ] Task: Conductor - User Manual Verification 'Phase 1: Architecture & Code Audit' (Protocol in workflow.md)

## Phase 2: Testing & Coverage
*Objective: Achieve >80% code coverage and ensure test reliability.*

- [ ] Task: Configure Coverage Reporting Tools
    - **Step:** Configure `pytest-cov` for Python backend.
    - **Step:** Configure frontend test runner (e.g., Vitest/Jest) for coverage.

- [ ] Task: Generate Initial Coverage Report
    - **Step:** Run full test suite with coverage enabled.
    - **Step:** Identify modules with low coverage (<80%).

- [ ] Task: Implement Missing Unit Tests (Backend)
    - **Step:** Write tests for identified low-coverage backend modules.
    - **Step:** Focus on `control` logic and `utils`.

- [ ] Task: Implement Missing Unit Tests (Frontend)
    - **Step:** Write tests for identified low-coverage frontend components.
    - **Step:** Focus on complex state management and API integration components.

- [ ] Task: Conductor - User Manual Verification 'Phase 2: Testing & Coverage' (Protocol in workflow.md)

## Phase 3: Performance Baselining
*Objective: Establish and verify the baseline performance for Centralized and Federated modes.*

- [ ] Task: Verify Centralized Training Pipeline
    - **Step:** Run a standardized centralized training experiment (short epochs).
    - **Step:** Verify metrics are logged to DB and sent via WebSocket.

- [ ] Task: Verify Federated Training Pipeline
    - **Step:** Run a standardized federated training experiment (2 clients, few rounds).
    - **Step:** Verify client aggregation and metric reporting.

- [ ] Task: Document Baseline Metrics
    - **Step:** Record execution time, resource usage, and final accuracy for both modes.
    - **Step:** Create a `BASELINE_REPORT.md` summarizing these findings.

- [ ] Task: Conductor - User Manual Verification 'Phase 3: Performance Baselining' (Protocol in workflow.md)
