# Module 1: Knowledge Base & Database (Updated)

**Agent Assignment:** backend-logic-architect
**Priority:** P0 (Start immediately)
**Dependencies:** None (reuses existing database)
**Estimated Effort:** 0.5-1 day (significantly reduced)

---

## Purpose

Provide persistent storage for research assistant experiments by **reusing the existing database infrastructure** (`runs`, `run_metrics` tables and CRUD operations).

**Key Change:** We do NOT create new tables. Instead, we create a lightweight adapter/repository layer on top of your existing `federated_pneumonia_detection.src.boundary` database.

---

## File Structure

```
federated_pneumonia_detection/src/control/agentic_systems/research_assistant/knowledge_base/
├── __init__.py
├── schemas.py              # Pydantic models (same as before)
├── repository.py           # NEW: Adapter for existing database
└── queries.py              # NEW: Helper queries for research assistant
```

**No changes needed to:**
- ❌ `boundary/models/` (no new tables)
- ❌ `boundary/CRUD/` (no new CRUD operations)
- ✅ Just add adapter layer to translate research concepts → existing schema

---

## How Research Assistant Uses Existing Schema

### Mapping Research Concepts → Database Tables

| Research Concept | Existing Table | How We Use It |
|------------------|----------------|---------------|
| **Research Session** | `runs` | One run per research session, `run_description` stores session config |
| **Experiment** | `run_metrics` (group by `step`) | Each `step` = one experiment with hyperparameters + results |
| **Hyperparameters** | `run_metrics` | Stored as metrics with prefix `hp_` (e.g., `hp_learning_rate`) |
| **Results** | `run_metrics` | Stored as metrics (`recall`, `accuracy`, `f1`, etc.) |
| **Agent Reasoning** | `run_metrics` | Stored as special metric `agent_reasoning` (truncated to float hash) |

### Example: Research Session → `runs` table

```python
# Create research session
run = run_crud.create(
    db,
    run_description=json.dumps({
        "type": "autonomous_research",
        "max_experiments": 30,
        "target_recall": 0.95,
        "paradigm": "both",
        "stopping_reason": None  # Filled when complete
    }),
    training_mode="both",  # or "centralized", "federated"
    status="in_progress",
    start_time=datetime.now()
)
```

### Example: Experiment → `run_metrics` (step-based)

```python
# Experiment #1: lr=0.001, batch=32, dropout=0.3
# Results: recall=0.933, accuracy=0.91

# Store hyperparameters
run_metric_crud.create(db, run_id=run.id, metric_name="hp_learning_rate", metric_value=0.001, step=1, dataset_type="config")
run_metric_crud.create(db, run_id=run.id, metric_name="hp_batch_size", metric_value=32.0, step=1, dataset_type="config")
run_metric_crud.create(db, run_id=run.id, metric_name="hp_dropout_rate", metric_value=0.3, step=1, dataset_type="config")

# Store results
run_metric_crud.create(db, run_id=run.id, metric_name="recall", metric_value=0.933, step=1, dataset_type="val")
run_metric_crud.create(db, run_id=run.id, metric_name="accuracy", metric_value=0.91, step=1, dataset_type="val")
run_metric_crud.create(db, run_id=run.id, metric_name="f1", metric_value=0.92, step=1, dataset_type="val")

# Store paradigm tested
run_metric_crud.create(db, run_id=run.id, metric_name="paradigm_centralized", metric_value=1.0, step=1, dataset_type="meta")
```

---

## Pydantic Models (schemas.py)

**Same as original design** - these are for type safety and API contracts:

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
    exploration_phase: str = Field(
        pattern="^(broad_exploration|smart_refinement|fine_tuning)$"
    )

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
    """Full experiment record (reconstructed from run_metrics)"""
    experiment_number: int  # step number
    paradigm: str
    hyperparameters: Dict[str, Any]
    metrics: Optional[Dict[str, float]]
    training_time_seconds: Optional[float]
    status: str
    error_message: Optional[str]
    agent_reasoning: str

class ResearchSession(BaseModel):
    """Research session metadata (from run + aggregated metrics)"""
    run_id: int
    start_time: datetime
    end_time: Optional[datetime]
    total_experiments: int
    best_centralized_recall: Optional[float]
    best_federated_recall: Optional[float]
    stopping_reason: Optional[str]
    config: Dict[str, Any]
```

---

## Repository Adapter (repository.py)

**Lightweight adapter to translate research assistant operations → existing CRUD:**

```python
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import json

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.engine import get_db

from .schemas import (
    ExperimentProposal,
    ExperimentResults,
    ExperimentRun,
    ResearchSession
)

class ResearchRepository:
    """
    Adapter to use existing database for research assistant

    Translates research assistant operations → existing run/run_metric CRUD
    """

    def __init__(self):
        self.db_generator = get_db()
        self.db: Session = next(self.db_generator)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    # Session Management

    def create_research_session(self, config: Dict[str, Any]) -> int:
        """
        Create a new research session (as a Run)

        Args:
            config: Session configuration (max_experiments, target_recall, etc.)

        Returns:
            run_id for the research session
        """
        run_description = json.dumps({
            "type": "autonomous_research",
            **config
        })

        run = run_crud.create(
            self.db,
            run_description=run_description,
            training_mode=config.get("paradigm", "both"),
            status="in_progress",
            start_time=datetime.now(),
            source_path=config.get("dataset_path")
        )

        self.db.commit()
        return run.id

    def end_research_session(self, run_id: int, stopping_reason: str):
        """Mark research session as complete"""

        # Get current run description and update with stopping reason
        run = run_crud.get(self.db, run_id)
        if run:
            config = json.loads(run.run_description)
            config["stopping_reason"] = stopping_reason

            run_crud.update(
                self.db,
                run_id,
                run_description=json.dumps(config),
                status="completed",
                end_time=datetime.now()
            )
            self.db.commit()

    def get_research_session(self, run_id: int) -> Optional[ResearchSession]:
        """Get research session metadata"""

        run = run_crud.get(self.db, run_id)
        if not run:
            return None

        config = json.loads(run.run_description)

        # Get experiment count (max step number)
        metrics = run_metric_crud.get_by_run(self.db, run_id)
        total_experiments = max([m.step for m in metrics], default=0)

        # Get best recalls
        best_cent = self._get_best_recall(run_id, "centralized")
        best_fed = self._get_best_recall(run_id, "federated")

        return ResearchSession(
            run_id=run.id,
            start_time=run.start_time,
            end_time=run.end_time,
            total_experiments=total_experiments,
            best_centralized_recall=best_cent,
            best_federated_recall=best_fed,
            stopping_reason=config.get("stopping_reason"),
            config=config
        )

    # Experiment CRUD

    def save_experiment(
        self,
        run_id: int,
        experiment_number: int,
        proposal: ExperimentProposal,
        results: ExperimentResults
    ) -> int:
        """
        Save experiment to database

        Args:
            run_id: Research session ID
            experiment_number: Experiment number (1, 2, 3, ...)
            proposal: Experiment configuration
            results: Experiment results

        Returns:
            experiment_number (same as input, for consistency)
        """

        # 1. Store hyperparameters as metrics
        for hp_name, hp_value in proposal.proposed_hyperparameters.items():
            run_metric_crud.create(
                self.db,
                run_id=run_id,
                metric_name=f"hp_{hp_name}",
                metric_value=float(hp_value),
                step=experiment_number,
                dataset_type="config"
            )

        # 2. Store paradigm
        paradigm_value = 1.0 if proposal.paradigm == "centralized" else 0.0
        run_metric_crud.create(
            self.db,
            run_id=run_id,
            metric_name="paradigm_centralized",
            metric_value=paradigm_value,
            step=experiment_number,
            dataset_type="meta"
        )

        # 3. Store results (if completed)
        if results.status == "completed" and results.metrics:
            for metric_name, metric_value in results.metrics.items():
                run_metric_crud.create(
                    self.db,
                    run_id=run_id,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    step=experiment_number,
                    dataset_type="val"
                )

            # Store training time
            run_metric_crud.create(
                self.db,
                run_id=run_id,
                metric_name="training_time_seconds",
                metric_value=results.training_time_seconds,
                step=experiment_number,
                dataset_type="meta"
            )

        # 4. Store status
        status_value = 1.0 if results.status == "completed" else 0.0
        run_metric_crud.create(
            self.db,
            run_id=run_id,
            metric_name="experiment_status_completed",
            metric_value=status_value,
            step=experiment_number,
            dataset_type="meta"
        )

        self.db.commit()
        return experiment_number

    def get_all_experiments(self, run_id: int) -> List[ExperimentRun]:
        """
        Get all experiments for a research session

        Returns experiments ordered by experiment_number
        """

        # Get all metrics for this run
        all_metrics = run_metric_crud.get_by_run(self.db, run_id)

        # Group by step (each step = one experiment)
        experiments_by_step = {}
        for metric in all_metrics:
            step = metric.step
            if step not in experiments_by_step:
                experiments_by_step[step] = []
            experiments_by_step[step].append(metric)

        # Reconstruct experiments
        experiments = []
        for step in sorted(experiments_by_step.keys()):
            experiment = self._reconstruct_experiment(step, experiments_by_step[step])
            experiments.append(experiment)

        return experiments

    def get_experiment_by_number(self, run_id: int, experiment_number: int) -> Optional[ExperimentRun]:
        """Get single experiment by number"""

        metrics = run_metric_crud.get_by_step(self.db, run_id, experiment_number)
        if not metrics:
            return None

        return self._reconstruct_experiment(experiment_number, metrics)

    def _reconstruct_experiment(self, step: int, metrics: List) -> ExperimentRun:
        """Reconstruct ExperimentRun from run_metrics"""

        hyperparameters = {}
        result_metrics = {}
        paradigm = "centralized"
        status = "completed"
        training_time = None

        for metric in metrics:
            # Hyperparameters (hp_ prefix)
            if metric.metric_name.startswith("hp_"):
                hp_name = metric.metric_name[3:]  # Remove "hp_" prefix
                hyperparameters[hp_name] = metric.metric_value

            # Paradigm
            elif metric.metric_name == "paradigm_centralized":
                paradigm = "centralized" if metric.metric_value == 1.0 else "federated"

            # Status
            elif metric.metric_name == "experiment_status_completed":
                status = "completed" if metric.metric_value == 1.0 else "failed"

            # Training time
            elif metric.metric_name == "training_time_seconds":
                training_time = metric.metric_value

            # Result metrics (dataset_type='val')
            elif metric.dataset_type == "val":
                result_metrics[metric.metric_name] = metric.metric_value

        return ExperimentRun(
            experiment_number=step,
            paradigm=paradigm,
            hyperparameters=hyperparameters,
            metrics=result_metrics if result_metrics else None,
            training_time_seconds=training_time,
            status=status,
            error_message=None,
            agent_reasoning=""  # Not stored (too verbose for float values)
        )

    # Analytics Queries

    def _get_best_recall(self, run_id: int, paradigm: str) -> Optional[float]:
        """Get best recall for a specific paradigm"""

        experiments = self.get_all_experiments(run_id)
        paradigm_exps = [e for e in experiments if e.paradigm == paradigm and e.metrics]

        if not paradigm_exps:
            return None

        recalls = [e.metrics.get("recall", 0) for e in paradigm_exps]
        return max(recalls) if recalls else None

    def get_session_summary(self, run_id: int) -> Dict[str, Any]:
        """Get session statistics"""

        experiments = self.get_all_experiments(run_id)
        completed = [e for e in experiments if e.status == "completed"]

        return {
            "total_experiments": len(experiments),
            "completed_experiments": len(completed),
            "failed_experiments": len(experiments) - len(completed),
            "best_centralized_recall": self._get_best_recall(run_id, "centralized") or 0.0,
            "best_federated_recall": self._get_best_recall(run_id, "federated") or 0.0,
            "avg_training_time": (
                sum(e.training_time_seconds for e in completed if e.training_time_seconds) / len(completed)
                if completed else 0.0
            )
        }


# Singleton instance
research_repository = ResearchRepository()
```

---

## Helper Queries (queries.py)

**Optional helper functions for common research assistant queries:**

```python
from typing import List, Tuple
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud

def get_hyperparameter_exploration_history(
    db: Session,
    run_id: int,
    parameter_name: str
) -> List[Tuple[float, float]]:
    """
    Get all tested values for a specific hyperparameter with their recalls

    Returns:
        List of (parameter_value, recall) tuples
    """

    # Get all steps that tested this hyperparameter
    hp_metrics = run_metric_crud.get_by_metric_name(db, run_id, f"hp_{parameter_name}")

    results = []
    for hp_metric in hp_metrics:
        # Get recall for this step
        recall_metric = run_metric_crud.get_by_step(db, run_id, hp_metric.step)
        recall = next((m.metric_value for m in recall_metric if m.metric_name == "recall"), None)

        if recall is not None:
            results.append((hp_metric.metric_value, recall))

    return results


def get_recent_experiments(db: Session, run_id: int, n: int = 5) -> List[int]:
    """
    Get last N experiment numbers (steps)

    Returns:
        List of step numbers (e.g., [23, 24, 25, 26, 27])
    """

    all_metrics = run_metric_crud.get_by_run(db, run_id)
    steps = sorted(set(m.step for m in all_metrics), reverse=True)

    return steps[:n]
```

---

## Testing Strategy

```python
# test_repository.py
import pytest
from .repository import ResearchRepository
from .schemas import ExperimentProposal, ExperimentResults

def test_create_research_session():
    """Test creating research session"""
    repo = ResearchRepository()

    run_id = repo.create_research_session({
        "max_experiments": 30,
        "target_recall": 0.95,
        "paradigm": "both"
    })

    assert run_id > 0

    session = repo.get_research_session(run_id)
    assert session.run_id == run_id
    assert session.config["max_experiments"] == 30

def test_save_and_retrieve_experiment():
    """Test saving and retrieving experiment"""
    repo = ResearchRepository()

    run_id = repo.create_research_session({"max_experiments": 5})

    proposal = ExperimentProposal(
        proposed_hyperparameters={"learning_rate": 0.001, "batch_size": 32},
        reasoning="Test",
        expected_outcome="Test",
        priority=5,
        paradigm="centralized",
        exploration_phase="broad_exploration"
    )

    results = ExperimentResults(
        metrics={"recall": 0.933, "accuracy": 0.91},
        training_time_seconds=120.0,
        status="completed"
    )

    repo.save_experiment(run_id, 1, proposal, results)

    experiment = repo.get_experiment_by_number(run_id, 1)

    assert experiment.experiment_number == 1
    assert experiment.paradigm == "centralized"
    assert experiment.hyperparameters["learning_rate"] == 0.001
    assert experiment.metrics["recall"] == 0.933

def test_get_session_summary():
    """Test session summary statistics"""
    repo = ResearchRepository()

    run_id = repo.create_research_session({})

    # Save multiple experiments
    for i in range(5):
        proposal = ExperimentProposal(...)
        results = ExperimentResults(metrics={"recall": 0.90 + i*0.01}, ...)
        repo.save_experiment(run_id, i+1, proposal, results)

    summary = repo.get_session_summary(run_id)

    assert summary["total_experiments"] == 5
    assert summary["best_centralized_recall"] > 0.9
```

---

## Benefits of This Approach

✅ **No database migrations** - Uses existing schema as-is
✅ **Minimal code** - Adapter layer only, ~200 lines vs 500+ for new tables
✅ **Reuses existing CRUD** - Leverages your well-tested operations
✅ **Consistent data model** - Research sessions stored alongside regular training runs
✅ **Existing UI compatibility** - Your current dashboard can query research runs
✅ **Easy to extend** - If you need new fields, just add new metric names

---

## Acceptance Criteria

- ✅ ResearchRepository successfully creates research sessions
- ✅ Experiments stored and retrieved with full fidelity
- ✅ Hyperparameters correctly encoded as `hp_*` metrics
- ✅ Session summary statistics accurate
- ✅ Works with both centralized and federated paradigms
- ✅ Unit tests achieve >90% coverage
- ✅ No changes required to existing boundary/CRUD code

---

**Status:** Ready for Implementation
**Blocked By:** None
**Blocks:** Module 2, Module 5
