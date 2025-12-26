# Comprehensive Refactoring Plan: runs_endpoints Module

**Date**: 2025-12-26
**Scope**: Refactor 3 endpoint files (runs_analytics.py, runs_list.py, runs_download.py)
**Goal**: Extract mixed logic into shared utilities following SOLID principles

---

## Executive Summary

Three endpoint files violate SRP by mixing:
- **runs_analytics.py** (114 lines): Routing + filtering + aggregation + statistics
- **runs_list.py** (328 lines): Routing + queries + metrics + federated logic + backfill
- **runs_download.py** (256 lines): Routing + serialization + file generation + formatting

**Common Issues Across All Files**:
1. Business logic embedded in endpoints
2. Duplicated metric extraction patterns
3. No response validation (Dict[str, Any] everywhere)
4. Database session management in endpoints
5. Federated vs centralized branching scattered
6. Response formatting inconsistencies

---

## Proposed Directory Structure

```
federated_pneumonia_detection/src/api/endpoints/runs_endpoints/
├── shared/                                    # NEW: Shared utilities (max grouping)
│   ├── __init__.py
│   ├── metrics.py                            # Extractors + Aggregators (all metric logic)
│   ├── summary_builder.py                    # Run summary construction + federated logic
│   ├── exporters.py                          # JSON/CSV/Text export + download service
│   ├── services.py                           # Backfill + ranking services
│   └── utils.py                              # All utilities (calculation + transformation)
├── runs_analytics.py                         # REFACTORED: Pure routing
├── runs_list.py                              # REFACTORED: Pure routing
├── runs_download.py                          # REFACTORED: Pure routing
├── runs_metrics.py
├── runs_server_evaluation.py
├── runs_federated_rounds.py
├── runs_debug.py
└── analytics_utils.py                        # DEPRECATED: Delete after migration

```

---

## Phase 1: Extract Schemas (Priority: Critical)

### Task 1.1: Create Response Schemas
**Location**: `src/api/endpoints/schema/runs_schemas.py` (already exists: schema/)

```python
# Analytics schemas
class ModeMetrics(BaseModel):
    count: int = Field(ge=0)
    avg_accuracy: Optional[float] = Field(None, ge=0, le=1)
    avg_precision: Optional[float] = Field(None, ge=0, le=1)
    avg_recall: Optional[float] = Field(None, ge=0, le=1)
    avg_f1: Optional[float] = Field(None, ge=0, le=1)
    avg_duration_minutes: Optional[float] = Field(None, ge=0)

class AnalyticsSummaryResponse(BaseModel):
    total_runs: int
    success_rate: float
    centralized: ModeMetrics
    federated: ModeMetrics
    top_runs: List[RunDetail]

# List schemas
class FederatedInfo(BaseModel):
    num_rounds: int
    num_clients: int
    has_server_evaluation: bool
    best_accuracy: Optional[float]
    best_recall: Optional[float]
    latest_round: Optional[int]
    latest_accuracy: Optional[float]

class RunSummary(BaseModel):
    id: int
    training_mode: str
    status: str
    start_time: Optional[str]
    end_time: Optional[str]
    best_val_recall: float
    metrics_count: int
    run_description: Optional[str]
    federated_info: Optional[FederatedInfo]

class RunsListResponse(BaseModel):
    runs: List[RunSummary]
    total: int

# Backfill schemas
class BackfillResult(BaseModel):
    run_id: int
    success: bool
    message: str
    rounds_processed: int
```

**Files Affected**:
- Create: `schema/runs_schemas.py` (NEW)
- Import in: `runs_analytics.py`, `runs_list.py`

**Acceptance Criteria**:
- All response schemas use Pydantic
- FastAPI OpenAPI docs auto-generated
- Response validation enforced

---

## Phase 2: Extract Metric Extraction (Priority: High)

### Task 2.1: Create Metric Extractor Strategy

**Problem**: Metric extraction duplicated 5+ times across files
- `runs_analytics.py` (analytics_utils.py:56-64, 121-143)
- `runs_list.py` (lines 37-44, 82-88)
- Mode branching (if federated... else...) in 3 places

**Solution**: Strategy pattern with interface

**Location**: `shared/extractors/`

```python
# base_extractor.py
from abc import ABC, abstractmethod
from typing import Dict, Optional

class MetricExtractor(ABC):
    """Abstract interface for metric extraction."""

    @abstractmethod
    def get_best_metric(
        self,
        db: Session,
        run_id: int,
        metric_name: str
    ) -> Optional[float]:
        """Get best value for specific metric."""
        pass

    def get_all_best_metrics(
        self,
        db: Session,
        run: Run
    ) -> Dict[str, Optional[float]]:
        """Get all standard metrics (accuracy, precision, recall, f1)."""
        return {
            'accuracy': self.get_best_metric(db, run.id, 'accuracy'),
            'precision': self.get_best_metric(db, run.id, 'precision'),
            'recall': self.get_best_metric(db, run.id, 'recall'),
            'f1_score': self.get_best_metric(db, run.id, 'f1_score'),
        }

# federated_extractor.py
class FederatedMetricExtractor(MetricExtractor):
    """Extract metrics from server evaluations."""

    def __init__(self, server_eval_crud):
        self.server_eval_crud = server_eval_crud

    def get_best_metric(self, db, run_id, metric_name):
        summary = self.server_eval_crud.get_summary_stats(db, run_id)
        # Handle nested dict access safely
        metric_key = f'best_{metric_name}'
        if summary.get(metric_key):
            return summary[metric_key].get('value')
        return None

    def get_training_rounds(self, db, run_id):
        """Get evaluations excluding round 0."""
        evals = self.server_eval_crud.get_by_run(db, run_id)
        return [e for e in evals if e.round_number > 0]

# centralized_extractor.py
class CentralizedMetricExtractor(MetricExtractor):
    """Extract metrics from run metrics table."""

    def __init__(self, run_metric_crud):
        self.run_metric_crud = run_metric_crud

    def get_best_metric(self, db, run_id, metric_name):
        metric = self.run_metric_crud.get_best_metric(
            db, run_id, f"val_{metric_name}"
        )
        return metric.metric_value if metric else None

# Factory function
def get_metric_extractor(run: Run) -> MetricExtractor:
    """Factory to get correct extractor based on training mode."""
    if run.training_mode == 'federated':
        return FederatedMetricExtractor(server_evaluation_crud)
    else:
        return CentralizedMetricExtractor(run_metric_crud)
```

**Files Affected**:
- Create: `shared/extractors/base_extractor.py`
- Create: `shared/extractors/federated_extractor.py`
- Create: `shared/extractors/centralized_extractor.py`
- Delete: Duplication in `analytics_utils.py` (lines 49-79, 115-143)
- Refactor: `runs_analytics.py`, `runs_list.py`

**Acceptance Criteria**:
- No if/else mode branching in endpoint files
- Extractor testable independently
- Can swap data sources without changing endpoints

---

## Phase 3: Extract Aggregation Logic (Priority: High)

### Task 3.1: Create Metrics Aggregator

**Problem**: Max/min/avg calculations duplicated across files

**Location**: `shared/aggregators/metrics_aggregator.py`

```python
class MetricsAggregator:
    """Calculate aggregate metrics from collections."""

    @staticmethod
    def get_best_metric(
        items: List,
        metric_field: str,
        default: Optional[float] = None
    ) -> Optional[float]:
        """Get best (max) value of metric field from items."""
        values = [
            getattr(item, metric_field)
            for item in items
            if getattr(item, metric_field, None) is not None
        ]
        return max(values) if values else default

    @staticmethod
    def get_worst_metric(items: List, metric_field: str) -> Optional[float]:
        """Get worst (min) value of metric field."""
        values = [
            getattr(item, metric_field)
            for item in items
            if getattr(item, metric_field, None) is not None
        ]
        return min(values) if values else None

    @staticmethod
    def get_latest_value(items: List, field: str) -> Optional[Any]:
        """Get value from last item (assumes ordered)."""
        return getattr(items[-1], field, None) if items else None

    @staticmethod
    def calculate_best_validation_recall(metrics: List) -> float:
        """Calculate best validation recall from RunMetrics."""
        val_recall_values = [
            m.metric_value
            for m in metrics
            if m.metric_name == "val_recall"
        ]
        return max(val_recall_values) if val_recall_values else 0.0
```

**Files Affected**:
- Create: `shared/aggregators/metrics_aggregator.py`
- Remove: Duplication in `runs_list.py` (lines 37-44, 82-88)
- Remove: Duplication in `analytics_utils.py` (lines 56-64, 121-128)

---

### Task 3.2: Create Run Aggregator

**Problem**: Statistics calculation in endpoint (analytics_utils.py:49-92)

**Location**: `shared/aggregators/run_aggregator.py`

```python
class RunAggregator:
    """Calculate aggregate statistics across multiple runs."""

    def __init__(self, metric_extractor: MetricExtractor):
        self.metric_extractor = metric_extractor

    def calculate_statistics(
        self,
        db: Session,
        runs: List[Run]
    ) -> Dict[str, Any]:
        """Calculate avg metrics and count for runs."""
        if not runs:
            return self._empty_stats()

        # Extract metrics for all runs
        all_metrics = []
        durations = []

        for run in runs:
            metrics = self.metric_extractor.get_all_best_metrics(db, run)
            all_metrics.append(metrics)

            duration = calculate_duration_minutes(run)
            if duration:
                durations.append(duration)

        # Calculate averages
        return {
            'count': len(runs),
            'avg_accuracy': self._safe_average([m['accuracy'] for m in all_metrics]),
            'avg_precision': self._safe_average([m['precision'] for m in all_metrics]),
            'avg_recall': self._safe_average([m['recall'] for m in all_metrics]),
            'avg_f1': self._safe_average([m['f1_score'] for m in all_metrics]),
            'avg_duration_minutes': self._safe_average(durations),
        }

    @staticmethod
    def _safe_average(values: List[Optional[float]]) -> Optional[float]:
        """Calculate average of non-None values."""
        non_none = [v for v in values if v is not None]
        return round(sum(non_none) / len(non_none), 4) if non_none else None

    @staticmethod
    def _empty_stats() -> Dict[str, Any]:
        return {
            'count': 0,
            'avg_accuracy': None,
            'avg_precision': None,
            'avg_recall': None,
            'avg_f1': None,
            'avg_duration_minutes': None,
        }
```

**Files Affected**:
- Create: `shared/aggregators/run_aggregator.py`
- Delete: `analytics_utils.py:calculate_mode_statistics()` (lines 49-92)
- Refactor: `runs_analytics.py` (use aggregator)

---

## Phase 4: Extract Export Logic (Priority: High)

### Task 4.1: Create Exporter Interface

**Problem**: Serialization logic embedded in endpoints (runs_download.py)

**Location**: `shared/exporters/`

```python
# base_exporter.py
from abc import ABC, abstractmethod

class DataExporter(ABC):
    """Abstract interface for data export."""

    @abstractmethod
    def export(self, data: Any) -> str:
        """Export data to string format."""
        pass

    @abstractmethod
    def get_media_type(self) -> str:
        """Return MIME type for this format."""
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """Return file extension (e.g., 'json', 'csv')."""
        pass

# json_exporter.py
import json
from datetime import datetime

class JSONExporter(DataExporter):
    """Export data as formatted JSON."""

    def export(self, data: Any) -> str:
        return json.dumps(data, indent=2, default=str)

    def get_media_type(self) -> str:
        return "application/json"

    def get_file_extension(self) -> str:
        return "json"

# csv_exporter.py
import csv
from io import StringIO

class CSVExporter(DataExporter):
    """Export training history as CSV."""

    def export(self, training_history: List[Dict]) -> str:
        if not training_history:
            return ""

        # Dynamic field discovery
        all_keys = set()
        for entry in training_history:
            all_keys.update(entry.keys())
        fieldnames = sorted(all_keys)

        # Write to StringIO
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_history)

        return output.getvalue()

    def get_media_type(self) -> str:
        return "text/csv"

    def get_file_extension(self) -> str:
        return "csv"

# text_report_exporter.py
class TextReportExporter(DataExporter):
    """Export as formatted text report."""

    def export(self, results: Dict) -> str:
        """Generate formatted text report."""
        sections = []

        # Experiment Info
        sections.append(self._format_experiment_info(results))

        # Final Metrics
        sections.append(self._format_final_metrics(results))

        # Training History
        sections.append(self._format_training_history(results))

        return "\n\n".join(sections)

    def _format_experiment_info(self, results: Dict) -> str:
        metadata = results.get('metadata', {})
        return f"""EXPERIMENT SUMMARY
==================
Experiment ID: {results.get('experiment_id', 'N/A')}
Status: {results.get('status', 'N/A')}
Start Time: {metadata.get('start_time', 'N/A')}
End Time: {metadata.get('end_time', 'N/A')}
Total Epochs: {metadata.get('total_epochs', 0)}"""

    # ... more formatting methods ...

    def get_media_type(self) -> str:
        return "text/plain"

    def get_file_extension(self) -> str:
        return "txt"
```

**Files Affected**:
- Create: `shared/exporters/base_exporter.py`
- Create: `shared/exporters/json_exporter.py`
- Create: `shared/exporters/csv_exporter.py`
- Create: `shared/exporters/text_report_exporter.py`
- Delete: Serialization logic from `runs_download.py` (lines 49, 100-117, 165-233)

---

### Task 4.2: Create Download Service

**Problem**: File generation + response building mixed in endpoint

**Location**: `shared/services/download_service.py`

```python
from datetime import datetime
from fastapi.responses import StreamingResponse

class DownloadService:
    """Orchestrate download preparation and response building."""

    def __init__(self, exporter: DataExporter):
        self.exporter = exporter

    def prepare_download(
        self,
        data: Any,
        run_id: int,
        prefix: str = "run"
    ) -> StreamingResponse:
        """Prepare data for download with proper headers."""
        # Export data
        content = self.exporter.export(data)

        # Generate filename
        filename = self._generate_filename(run_id, prefix)

        # Build response
        return StreamingResponse(
            iter([content]),
            media_type=self.exporter.get_media_type(),
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": self.exporter.get_media_type(),
            }
        )

    def _generate_filename(self, run_id: int, prefix: str) -> str:
        """Generate timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = self.exporter.get_file_extension()
        return f"{prefix}_{run_id}_metrics_{timestamp}.{extension}"
```

**Files Affected**:
- Create: `shared/services/download_service.py`
- Delete: Response building from `runs_download.py` (lines 56-63, 124-131, 240-247)

---

## Phase 5: Extract Builders & Formatters (Priority: Medium)

### Task 5.1: Create Run Summary Builder

**Problem**: Summary construction scattered in `runs_list.py` (lines 46-110)

**Location**: `shared/builders/run_summary_builder.py`

```python
class RunSummaryBuilder:
    """Build run summaries with mode-specific details."""

    def __init__(self, metrics_aggregator, federated_summarizer=None):
        self.metrics_aggregator = metrics_aggregator
        self.federated_summarizer = federated_summarizer

    def build(self, run: Run, db: Session) -> Dict[str, Any]:
        """Build complete run summary."""
        # Base summary (common to all modes)
        summary = self._build_base_summary(run)

        # Add mode-specific info
        if run.training_mode == "federated" and self.federated_summarizer:
            summary["federated_info"] = self.federated_summarizer.summarize(run, db)
        else:
            summary["federated_info"] = None

        return summary

    def _build_base_summary(self, run: Run) -> Dict[str, Any]:
        """Build base summary fields."""
        best_val_recall = self.metrics_aggregator.calculate_best_validation_recall(
            run.metrics
        )

        return {
            "id": run.id,
            "training_mode": run.training_mode,
            "status": run.status,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "best_val_recall": best_val_recall,
            "metrics_count": len(run.metrics) if run.metrics else 0,
            "run_description": run.run_description,
        }

# Federated-specific summarizer
class FederatedRunSummarizer:
    """Handle federated-specific summary logic."""

    def __init__(self, server_eval_crud, metrics_aggregator):
        self.server_eval_crud = server_eval_crud
        self.metrics_aggregator = metrics_aggregator

    def summarize(self, run: Run, db: Session) -> Dict[str, Any]:
        """Summarize federated run with server evaluations."""
        training_rounds = self._get_training_rounds(run, db)
        num_clients = len(run.clients) if run.clients else 0

        if not training_rounds:
            return {
                "num_rounds": 0,
                "num_clients": num_clients,
                "has_server_evaluation": False,
            }

        return {
            "num_rounds": len(training_rounds),
            "num_clients": num_clients,
            "has_server_evaluation": True,
            "best_accuracy": self.metrics_aggregator.get_best_metric(
                training_rounds, "accuracy"
            ),
            "best_recall": self.metrics_aggregator.get_best_metric(
                training_rounds, "recall"
            ),
            "latest_round": training_rounds[-1].round_number,
            "latest_accuracy": training_rounds[-1].accuracy,
        }

    def _get_training_rounds(self, run: Run, db: Session):
        """Get evaluations excluding round 0."""
        evals = self.server_eval_crud.get_by_run(db, run.id)
        return [e for e in evals if e.round_number > 0]
```

**Files Affected**:
- Create: `shared/builders/run_summary_builder.py`
- Delete: Summary logic from `runs_list.py` (lines 46-110)

---

### Task 5.2: Create Utility Functions

**Location**: `shared/utils/`

```python
# calculation_utils.py
from typing import Optional
from datetime import datetime

def calculate_duration_minutes(run: Run) -> Optional[float]:
    """Calculate training duration in minutes."""
    if not run.start_time or not run.end_time:
        return None
    return round((run.end_time - run.start_time).total_seconds() / 60, 2)

def calculate_filtering_coverage(
    total_filtered: int,
    total_unfiltered: int
) -> float:
    """Calculate % of unfiltered runs that matched filters."""
    return total_filtered / total_unfiltered if total_unfiltered > 0 else 0.0

# data_utils.py
from typing import Dict, List, Optional, Any

def safe_get_nested(
    dict_obj: Dict,
    path: List[str],
    default=None
) -> Any:
    """Safely navigate nested dicts. Path=['a', 'b'] → dict_obj['a']['b']"""
    value = dict_obj
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
    return value if value is not None else default

def safe_average(values: List[Optional[float]]) -> Optional[float]:
    """Calculate average of non-None values."""
    non_none = [v for v in values if v is not None]
    return round(sum(non_none) / len(non_none), 4) if non_none else None
```

**Files Affected**:
- Create: `shared/utils/calculation_utils.py`
- Create: `shared/utils/data_utils.py`
- Move: Functions from `analytics_utils.py` (lines 171-183)
- Remove: Duration calc duplication (analytics_utils.py:82-84, 152-153)

---

### Task 5.3: Move Transformation Utils

**Problem**: `utils.py` (214 lines) needs to move to `shared/utils/transformation_utils.py`

**Location**: `shared/utils/transformation_utils.py`

```python
# Move entire utils.py contents here
# Keep: _calculate_summary_statistics, _transform_run_to_results, _find_best_epoch
```

**Files Affected**:
- Move: `utils.py` → `shared/utils/transformation_utils.py`
- Update imports in: `runs_download.py` (line 13)

---

## Phase 6: Extract Services (Priority: Medium)

### Task 6.1: Create Backfill Service

**Problem**: Backfill workflow in endpoint (`runs_list.py`:224-318)

**Location**: `shared/services/backfill_service.py`

```python
import json
import ast
from pathlib import Path
from typing import Dict, Any

class BackfillService:
    """Orchestrate backfill of server evaluations from JSON."""

    def __init__(self, run_crud, server_eval_crud, logger):
        self.run_crud = run_crud
        self.server_eval_crud = server_eval_crud
        self.logger = logger

    def backfill_from_json(
        self,
        db: Session,
        run_id: int
    ) -> Dict[str, Any]:
        """Load and persist evaluations from JSON file."""
        # Validate run exists
        run = self.run_crud.get(db, run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        # Load JSON
        json_file = Path(f"results_{run_id}.json")
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract evaluations
        server_evals = data.get("evaluate_metrics_serverapp", {})
        if not server_evals:
            return {
                "run_id": run_id,
                "success": False,
                "message": "No server evaluations found",
                "rounds_processed": 0,
            }

        # Process rounds
        rounds_processed = 0
        try:
            for round_num_str, metric_str in server_evals.items():
                round_num = int(round_num_str)

                # Parse metrics
                metrics_dict = ast.literal_eval(metric_str)
                extracted_metrics = self._extract_metrics(metrics_dict)

                # Persist
                self.server_eval_crud.create_evaluation(
                    db=db,
                    run_id=run_id,
                    round_number=round_num,
                    metrics=extracted_metrics,
                    num_samples=metrics_dict.get("num_samples"),
                )
                rounds_processed += 1

            db.commit()
            self.logger.info(
                f"[Backfill] SUCCESS: {rounds_processed} evaluations"
            )

            return {
                "run_id": run_id,
                "success": True,
                "message": f"Backfilled {rounds_processed} evaluations",
                "rounds_processed": rounds_processed,
            }

        except Exception as e:
            db.rollback()
            self.logger.error(f"[Backfill] ERROR: {e}")
            raise

    @staticmethod
    def _extract_metrics(metrics_dict: Dict) -> Dict[str, float]:
        """Extract and normalize metric names."""
        return {
            "loss": metrics_dict.get("server_loss", 0.0),
            "accuracy": metrics_dict.get("server_accuracy"),
            "precision": metrics_dict.get("server_precision"),
            "recall": metrics_dict.get("server_recall"),
            "f1_score": metrics_dict.get("server_f1"),
            "auroc": metrics_dict.get("server_auroc"),
        }
```

**Files Affected**:
- Create: `shared/services/backfill_service.py`
- Delete: Backfill logic from `runs_list.py` (lines 235-307)
- Endpoint becomes: Validate input → Call service → Return result

---

### Task 6.2: Create Ranking Service

**Problem**: Top-N selection in endpoint (`runs_analytics.py`:85-89)

**Location**: `shared/services/ranking_service.py`

```python
from typing import List, Dict, Any

class RunRanker:
    """Rank and filter runs by metrics."""

    @staticmethod
    def get_top_runs(
        runs: List[Dict[str, Any]],
        metric: str = "best_accuracy",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return top N runs sorted by metric descending."""
        return sorted(
            runs,
            key=lambda x: x.get(metric, 0.0),
            reverse=True
        )[:limit]
```

**Files Affected**:
- Create: `shared/services/ranking_service.py`
- Remove: Sorting logic from `runs_analytics.py` (lines 85-89)

---

## Phase 7: Refactor Endpoints (Priority: High - Final Step)

### Task 7.1: Refactor runs_analytics.py

**Before**: 114 lines with mixed logic
**After**: ~25-30 lines, pure orchestration

```python
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime, timedelta

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

from ..schema.runs_schemas import AnalyticsSummaryResponse
from .shared.extractors.base_extractor import get_metric_extractor
from .shared.aggregators.run_aggregator import RunAggregator
from .shared.services.ranking_service import RunRanker

router = APIRouter()
logger = get_logger(__name__)


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary(
    status: Optional[str] = Query("completed"),
    training_mode: Optional[str] = Query(None),
    days: Optional[int] = Query(None)
):
    """Get aggregated training statistics."""
    db = get_session()

    try:
        # Fetch runs (DB-level filtering)
        runs = run_crud.get_by_status_mode_and_date(
            db, status=status, training_mode=training_mode, days=days
        )

        if not runs:
            return AnalyticsSummaryResponse.empty()

        # Split by mode
        centralized_runs = [r for r in runs if r.training_mode == "centralized"]
        federated_runs = [r for r in runs if r.training_mode == "federated"]

        # Calculate statistics
        centralized_aggregator = RunAggregator(get_metric_extractor("centralized"))
        federated_aggregator = RunAggregator(get_metric_extractor("federated"))

        centralized_stats = centralized_aggregator.calculate_statistics(db, centralized_runs)
        federated_stats = federated_aggregator.calculate_statistics(db, federated_runs)

        # Get top runs
        all_run_details = [extract_run_details(db, r) for r in runs]
        top_runs = RunRanker.get_top_runs(all_run_details, metric="best_accuracy")

        return AnalyticsSummaryResponse(
            total_runs=len(runs),
            success_rate=calculate_filtering_coverage(len(runs), ...),
            centralized=centralized_stats,
            federated=federated_stats,
            top_runs=top_runs
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
```

**Files Affected**:
- Refactor: `runs_analytics.py` (delete analytics_utils.py usage)
- Delete: `analytics_utils.py` (203 lines → split into services)

---

### Task 7.2: Refactor runs_list.py

**Before**: 328 lines with 4 endpoints + federated logic + backfill
**After**: ~80-100 lines, pure routing

```python
# /list endpoint becomes:
@router.get("/list", response_model=RunsListResponse)
async def list_all_runs():
    """List all runs with summaries."""
    db = get_session()

    try:
        runs = run_crud.get_all_ordered_by_start_time(db)

        summary_builder = RunSummaryBuilder(
            metrics_aggregator=MetricsAggregator(),
            federated_summarizer=FederatedRunSummarizer(server_evaluation_crud, MetricsAggregator())
        )

        run_summaries = [summary_builder.build(run, db) for run in runs]

        return RunsListResponse(runs=run_summaries, total=len(run_summaries))

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# /backfill endpoint becomes:
@router.post("/backfill/{run_id}/server-evaluations", response_model=BackfillResult)
async def backfill_server_evaluations(run_id: int):
    """Backfill evaluations from JSON."""
    db = get_session()

    try:
        backfill_service = BackfillService(run_crud, server_evaluation_crud, logger)
        result = backfill_service.backfill_from_json(db, run_id)
        return BackfillResult(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
```

**Files Affected**:
- Refactor: `runs_list.py` (remove federated logic, backfill logic)
- Use: `RunSummaryBuilder`, `FederatedRunSummarizer`, `BackfillService`

---

### Task 7.3: Refactor runs_download.py

**Before**: 256 lines with serialization + file gen
**After**: ~40-50 lines, pure routing

```python
# /download/json endpoint becomes:
@router.get("/download/json/{run_id}")
async def download_json(run_id: int):
    """Download run as JSON."""
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)
        if not run:
            raise HTTPException(404, "Run not found")

        results = transform_run_to_results(run)  # From transformation_utils

        download_service = DownloadService(JSONExporter())
        return download_service.prepare_download(results, run_id, prefix="run")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))
    finally:
        db.close()

# /download/csv endpoint becomes:
@router.get("/download/csv/{run_id}")
async def download_csv(run_id: int):
    """Download training history as CSV."""
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)
        if not run:
            raise HTTPException(404, "Run not found")

        results = transform_run_to_results(run)
        training_history = results.get("training_history", [])

        if not training_history:
            raise HTTPException(404, "No training history")

        download_service = DownloadService(CSVExporter())
        return download_service.prepare_download(training_history, run_id, prefix="training_history")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))
    finally:
        db.close()

# /download/summary endpoint becomes similar (TextReportExporter)
```

**Files Affected**:
- Refactor: `runs_download.py` (remove serialization, file gen)
- Use: `DownloadService`, `JSONExporter`, `CSVExporter`, `TextReportExporter`

---

## File Size Compliance Check

| File | Current | After Refactor | Target (<150) | ✓ |
|------|---------|----------------|---------------|---|
| runs_analytics.py | 114 | ~25-30 | 150 | ✓ |
| runs_list.py | 328 | ~80-100 | 150 | ✓ |
| runs_download.py | 256 | ~40-50 | 150 | ✓ |
| analytics_utils.py | 203 | DELETE | - | ✓ |
| **New: base_extractor.py** | - | ~40 | 150 | ✓ |
| **New: federated_extractor.py** | - | ~50 | 150 | ✓ |
| **New: centralized_extractor.py** | - | ~30 | 150 | ✓ |
| **New: metrics_aggregator.py** | - | ~70 | 150 | ✓ |
| **New: run_aggregator.py** | - | ~90 | 150 | ✓ |
| **New: run_summary_builder.py** | - | ~120 | 150 | ✓ |
| **New: json_exporter.py** | - | ~30 | 150 | ✓ |
| **New: csv_exporter.py** | - | ~50 | 150 | ✓ |
| **New: text_report_exporter.py** | - | ~140 | 150 | ✓ |
| **New: download_service.py** | - | ~50 | 150 | ✓ |
| **New: backfill_service.py** | - | ~100 | 150 | ✓ |
| **New: ranking_service.py** | - | ~20 | 150 | ✓ |
| **New: calculation_utils.py** | - | ~30 | 150 | ✓ |
| **New: data_utils.py** | - | ~40 | 150 | ✓ |
| **Moved: transformation_utils.py** | 214 | 214 | 150 | ✗ (split later) |

---

## Implementation Sequence

### Week 1: Foundation (Phases 1-2)
- Day 1-2: Create schemas (runs_schemas.py)
- Day 3-5: Create extractors (base, federated, centralized)

### Week 2: Core Logic (Phases 3-4)
- Day 1-3: Create aggregators (metrics, runs)
- Day 4-7: Create exporters (JSON, CSV, text)

### Week 3: Services & Builders (Phases 5-6)
- Day 1-3: Create builders (summary, response)
- Day 4-7: Create services (download, backfill, ranking)

### Week 4: Refactor Endpoints (Phase 7)
- Day 1-2: Refactor runs_analytics.py
- Day 3-5: Refactor runs_list.py
- Day 6-7: Refactor runs_download.py

### Week 5: Testing & Documentation
- Day 1-3: Unit tests for all extractors/services
- Day 4-5: Integration tests for endpoints
- Day 6-7: Documentation + cleanup

---

## Critical Success Factors

1. **No Breaking Changes**: API interfaces must remain identical
2. **SOLID Compliance**: Each file has one responsibility
3. **File Size**: All files <150 lines
4. **Testability**: Services can be tested independently
5. **Type Safety**: Use Pydantic for all responses
6. **No Duplication**: Shared logic extracted once

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking API changes | Create integration tests before refactoring |
| Incorrect metric extraction | Unit test extractors with fixtures |
| Missing edge cases | Code review + QA testing |
| Import circular dependencies | Use dependency injection |
| Performance regression | Benchmark before/after |

---

## Acceptance Criteria

- [ ] All 3 endpoint files <150 lines
- [ ] All new modules <150 lines
- [ ] Pydantic schemas for all responses
- [ ] No duplicated metric extraction logic
- [ ] No if/else mode branching in endpoints
- [ ] All services unit testable
- [ ] API behavior unchanged (existing tests pass)
- [ ] OpenAPI docs auto-generated

---

**Next Steps**:
1. Review this plan
2. Get user approval
3. Create `shared/` directory structure
4. Begin Phase 1 (schemas)
