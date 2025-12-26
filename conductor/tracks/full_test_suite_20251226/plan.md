# Track Plan: Full Functional Test Suite (100% Coverage)

### Phase 1: Test Infrastructure & Foundation (Entities/Utils)
*Objective: Set up global fixtures and achieve 100% coverage for core domain objects and utilities.*

- [x] **Task: Global Test Environment & Fixtures Setup** be538d7
    - [ ] Create shared Pytest fixtures for mock medical datasets (CSVs and dummy images).
    - [ ] Set up specialized Vitest mocks for browser-only APIs (WebSocket, File API).
    - [ ] Configure coverage tools to fail if thresholds drop below 100%.
- [x] **Task: Unit Tests for Entities (`src/entities`)** 1c0f260
    - [ ] Test `ResNetWithCustomHead` architecture, forward pass, and layer freezing.
    - [ ] Test `CustomImageDataset` with various label scenarios and image transforms.
    - [ ] Validate model weight saving/loading logic.
- [x] **Task: Unit Tests for Utilities (`src/utils`)** 70872f5
    - [ ] Test `ConfigLoader` with YAML overrides and environment variable injections.
    - [ ] Test `ImageTransforms` for consistency across RGB and Grayscale modes.
    - [ ] Test data processing functions for edge cases (missing labels, corrupt metadata).
- [~] **Task: Conductor - User Manual Verification 'Phase 1: Test Infrastructure & Foundation' (Protocol in workflow.md)**

### Phase 2: Backend Logic & Communication (Control/Boundary)
*Objective: 100% coverage for trainers, orchestration logic, and API/WebSocket boundaries.*

- [ ] **Task: Unit & Integration Tests for Control Layer (`src/control`)**
    - [ ] **Unit:** Mock `DataLoader` to test `CentralizedTrainer` epoch logic in isolation.
    - [ ] **Unit:** Test `FederatedTrainer` client selection and weight aggregation math.
    - [ ] **Integration:** Test `ExperimentOrchestrator` flow between Centralized and Federated modes.
    - [ ] **Unit:** Test RAG/Arxiv integration agents with mocked external API responses.
- [ ] **Task: Unit & Integration Tests for Boundary Layer (`src/api`)**
    - [ ] **Unit:** Test individual Pydantic schemas and validation logic.
    - [ ] **Integration:** Test FastAPI endpoints for all CRUD operations using `TestClient`.
    - [ ] **Integration:** Validate WebSocket message broadcasting and client connection handling.
    - [ ] **Unit:** Test `RunDAO` and `MetricDAO` with a mocked SQL database.
- [ ] **Task: Conductor - User Manual Verification 'Phase 2: Backend Logic & Communication' (Protocol in workflow.md)**

### Phase 3: Frontend Services & Hooks
*Objective: Robust testing of the API client, WebSocket state, and custom logic.*

- [ ] **Task: Unit Tests for Services & Types**
    - [ ] Test `api.ts` client logic (request formatting, error status handling).
    - [ ] Test `websocket.ts` for auto-reconnect logic and event dispatcher.
    - [ ] Test `configMapper.ts` to ensure frontend UI state maps 1:1 to backend schemas.
- [ ] **Task: Unit Tests for Custom Hooks**
    - [ ] Test `useWebSocket` for state updates during incoming training metrics.
    - [ ] Test UI state hooks (e.g., config toggles) for predictable behavior.
- [ ] **Task: Conductor - User Manual Verification 'Phase 3: Frontend Services & Hooks' (Protocol in workflow.md)**

### Phase 4: Frontend Components & Integration
*Objective: Achieve 100% coverage for UI components and cross-component workflows.*

- [ ] **Task: Component Unit Tests**
    - [ ] Test `ExperimentConfig` for all 30+ parameters and collapsible section logic.
    - [ ] Test `DatasetUpload` for file validation messages and progress states.
    - [ ] Test `TrainingExecution` for real-time progress bar rendering.
    - [ ] Test `ResultsVisualization` with complex charting data mocks.
- [ ] **Task: Frontend Integration Tests**
    - [ ] Test the handover from `DatasetUpload` to `ExperimentConfig` using a shared context.
    - [ ] Validate that training completion correctly triggers the transition to results view.
- [ ] **Task: Conductor - User Manual Verification 'Phase 4: Frontend Components & Integration' (Protocol in workflow.md)**

### Phase 5: System Verification & Quality Audit
*Objective: Final E2E flow and verification of 100% coverage across the repository.*

- [ ] **Task: End-to-End Functional Tests**
    - [ ] Script a full headless simulation: Upload -> Configure -> Train -> View Results.
    - [ ] Verify that final metrics in the DB match the metrics displayed in the UI during E2E.
- [ ] **Task: Final Static Analysis & Coverage Audit**
    - [ ] Run `mypy` and `tsc` to verify absolute type integrity.
    - [ ] Generate final HTML coverage reports for both Backend and Frontend.
    - [ ] Perform a "Dead Code" sweep to remove any unreachable lines preventing 100% coverage.
- [ ] **Task: Conductor - User Manual Verification 'Phase 5: System Verification & Quality Audit' (Protocol in workflow.md)**
