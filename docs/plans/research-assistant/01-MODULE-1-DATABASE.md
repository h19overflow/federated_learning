# Module 1: Knowledge Base & Database

**Agent Assignment:** backend-logic-architect
**Priority:** P0 (Start immediately)
**Dependencies:** None
**Estimated Effort:** 1-2 days

---

## Purpose

Provide persistent storage for all experiment runs, results, and research sessions. Acts as the "memory" for the research assistant.

---

## File Structure

```
federated_pneumonia_detection/src/control/agentic_systems/research_assistant/knowledge_base/
├── __init__.py
├── schemas.py              # Pydantic models
├── database.py             # SQLAlchemy models + CRUD operations
└── migrations/             # Database migrations (Alembic)
    └── versions/
```

---

## Database Schema

### Tables

**1. research_sessions**
```sql
CREATE TABLE research_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    total_experiments INTEGER DEFAULT 0,
    best_centralized_recall REAL,
    best_federated_recall REAL,
    stopping_reason TEXT,
    config JSON  -- max_experiments, target_recall, etc.
);
```

**2. experiments**
```sql
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    experiment_number INTEGER NOT NULL,  -- 1, 2, 3...
    timestamp TIMESTAMP NOT NULL,
    paradigm TEXT NOT NULL,  -- "centralized" or "federated"
    status TEXT NOT NULL,  -- "completed", "failed", "running"

    -- Hyperparameters (JSON)
    hyperparameters JSON NOT NULL,

    -- Results (JSON)
    metrics JSON,  -- {recall, accuracy, f1, auroc, confusion_matrix}

    -- Metadata
    training_time_seconds REAL,
    error_message TEXT,
    agent_reasoning TEXT,  -- Why this experiment was proposed

    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
);
```

**3. hyperparameter_ranges**
```sql
CREATE TABLE hyperparameter_ranges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parameter_name TEXT NOT NULL,
    min_value REAL,
    max_value REAL,
    suggested_values JSON,  -- List of discrete values to try
    explored_count INTEGER DEFAULT 0
);
```

---

## Pydantic Models (schemas.py)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List

class ExperimentProposal(BaseModel):
    """Input from Module 2 (Orchestrator)"""
    proposed_hyperparameters: Dict[str, Any] = Field(
        description="Hyperparameters to test",
        example={"learning_rate": 0.001, "batch_size": 32, "dropout_rate": 0.3}
    )
    reasoning: str = Field(description="Why these hyperparameters were chosen")
    expected_outcome: str = Field(description="What we expect to learn")
    priority: int = Field(ge=1, le=10, description="Priority level")
    paradigm: str = Field(pattern="^(centralized|federated)$")

class ExperimentResults(BaseModel):
    """Output from Module 3 (Browser Automation)"""
    metrics: Dict[str, float] = Field(
        description="Training metrics",
        example={
            "recall": 0.933,
            "accuracy": 0.91,
            "precision": 0.89,
            "f1": 0.91,
            "auroc": 0.95
        }
    )
    confusion_matrix: Optional[Dict[str, int]] = Field(
        description="TP, TN, FP, FN counts",
        example={"tp": 450, "tn": 520, "fp": 30, "fn": 20}
    )
    training_time_seconds: float
    status: str = Field(pattern="^(completed|failed)$")
    error_message: Optional[str] = None

class ExperimentRun(BaseModel):
    """Full experiment record (stored in DB)"""
    id: int
    session_id: int
    experiment_number: int
    timestamp: datetime
    paradigm: str
    hyperparameters: Dict[str, Any]
    metrics: Optional[Dict[str, float]]
    confusion_matrix: Optional[Dict[str, int]]
    training_time_seconds: Optional[float]
    status: str
    error_message: Optional[str]
    agent_reasoning: str

class ResearchSession(BaseModel):
    """Research session metadata"""
    id: int
    start_time: datetime
    end_time: Optional[datetime]
    total_experiments: int
    best_centralized_recall: Optional[float]
    best_federated_recall: Optional[float]
    stopping_reason: Optional[str]
    config: Dict[str, Any]
```

---

## CRUD Operations (database.py)

```python
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from datetime import datetime
import json

class ExperimentDatabase:
    def __init__(self, db_url: str = "sqlite:///research_assistant.db"):
        self.engine = create_engine(db_url)
        self._create_tables()

    # Session Management
    def create_session(self, config: dict) -> int:
        """Start a new research session"""
        # Returns session_id
        pass

    def end_session(self, session_id: int, stopping_reason: str):
        """Mark session as complete"""
        pass

    # Experiment CRUD
    def save_experiment(
        self,
        session_id: int,
        proposal: ExperimentProposal,
        results: ExperimentResults
    ) -> int:
        """Save completed experiment, returns experiment_id"""
        pass

    def get_all_experiments(self, session_id: int) -> List[ExperimentRun]:
        """Get all experiments for a session, ordered by timestamp"""
        pass

    def get_experiment_by_id(self, experiment_id: int) -> ExperimentRun:
        """Get single experiment"""
        pass

    def update_experiment_status(self, experiment_id: int, status: str, error_msg: str = None):
        """Update experiment status (for tracking running experiments)"""
        pass

    # Analytics Queries
    def get_best_result(self, session_id: int, paradigm: str, metric: str = "recall") -> ExperimentRun:
        """Get best experiment for a given paradigm and metric"""
        pass

    def get_experiments_by_paradigm(self, session_id: int, paradigm: str) -> List[ExperimentRun]:
        """Filter experiments by paradigm"""
        pass

    def get_hyperparameter_exploration_history(
        self,
        session_id: int,
        parameter_name: str
    ) -> List[tuple]:
        """Get all tested values for a specific hyperparameter
        Returns: [(value, recall), ...]
        """
        pass

    def get_recent_experiments(self, session_id: int, n: int = 5) -> List[ExperimentRun]:
        """Get last N experiments (for convergence detection)"""
        pass

    def get_session_summary(self, session_id: int) -> dict:
        """Get session statistics"""
        # Returns: {
        #   "total_experiments": 25,
        #   "best_centralized_recall": 0.947,
        #   "best_federated_recall": 0.931,
        #   "avg_training_time": 120.5,
        #   "failed_experiments": 2
        # }
        pass
```

---

## API Endpoints (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI()

@app.post("/sessions", response_model=int)
def create_research_session(config: dict):
    """Start new research session"""
    pass

@app.get("/sessions/{session_id}/experiments", response_model=List[ExperimentRun])
def get_session_experiments(session_id: int):
    """Get all experiments for a session"""
    pass

@app.post("/experiments", response_model=int)
def save_experiment(
    session_id: int,
    proposal: ExperimentProposal,
    results: ExperimentResults
):
    """Save experiment results"""
    pass

@app.get("/experiments/{experiment_id}", response_model=ExperimentRun)
def get_experiment(experiment_id: int):
    """Get experiment by ID"""
    pass

@app.get("/sessions/{session_id}/best")
def get_best_experiments(session_id: int, paradigm: str = None):
    """Get best results for session"""
    pass

@app.get("/sessions/{session_id}/summary")
def get_session_summary(session_id: int):
    """Get session statistics"""
    pass
```

---

## Testing Strategy

**Unit Tests:**
```python
# test_database.py
def test_create_session():
    db = ExperimentDatabase(":memory:")
    session_id = db.create_session({"max_experiments": 30})
    assert session_id > 0

def test_save_and_retrieve_experiment():
    db = ExperimentDatabase(":memory:")
    session_id = db.create_session({})

    proposal = ExperimentProposal(
        proposed_hyperparameters={"lr": 0.001},
        reasoning="Test",
        expected_outcome="Test",
        priority=5,
        paradigm="centralized"
    )
    results = ExperimentResults(
        metrics={"recall": 0.93},
        training_time_seconds=120.0,
        status="completed"
    )

    exp_id = db.save_experiment(session_id, proposal, results)
    retrieved = db.get_experiment_by_id(exp_id)

    assert retrieved.metrics["recall"] == 0.93
    assert retrieved.paradigm == "centralized"

def test_get_best_result():
    # Test with multiple experiments, verify best is returned
    pass

def test_hyperparameter_exploration_history():
    # Test tracking of parameter values over time
    pass
```

---

## Acceptance Criteria

- ✅ Database schema supports all required fields
- ✅ CRUD operations work correctly with Pydantic models
- ✅ Can store and retrieve experiments with full fidelity
- ✅ Analytics queries return correct results
- ✅ FastAPI endpoints functional and documented
- ✅ Unit tests achieve >90% coverage
- ✅ Database migrations work (Alembic)

---

## Notes for Implementation

1. Use SQLAlchemy for ORM (easier testing)
2. Support both SQLite (dev) and PostgreSQL (production)
3. Add indexes on frequently queried columns (session_id, paradigm, timestamp)
4. JSON fields for flexibility (hyperparameters, metrics)
5. Consider adding a `tags` field for categorizing experiments

---

**Status:** Ready for Implementation
**Blocked By:** None
**Blocks:** Module 2, Module 5
