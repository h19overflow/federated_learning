import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.models import (
    Run,
    RunMetric,
    ServerEvaluation,
)


class RunCRUD(BaseCRUD[Run]):
    """CRUD operations for Run model."""

    def __init__(self):
        super().__init__(Run)
        self.logger = logging.getLogger(__name__)

    def get_by_experiment(self, db: Session, experiment_id: int) -> List[Run]:
        """Get all runs for a specific experiment."""
        return (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.experiment_id == experiment_id)
            .all()
        )

    def get_by_status(self, db: Session, status: str) -> List[Run]:
        """Get runs by status."""
        return (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.status == status)
            .all()
        )

    def get_by_training_mode(self, db: Session, training_mode: str) -> List[Run]:
        """Get runs by training mode."""
        return (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.training_mode == training_mode)
            .all()
        )

    def get_with_config(self, db: Session, id: int) -> Optional[Run]:
        """Get run with its configuration."""
        return (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.id == id)
            .first()
        )

    def get_with_metrics(self, db: Session, id: int) -> Optional[Run]:
        """Get run with all its metrics."""
        return (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.id == id)
            .first()
        )

    def get_with_artifacts(self, db: Session, id: int) -> Optional[Run]:
        """Get run with all its artifacts."""
        return (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.id == id)
            .first()
        )

    def update_status(self, db: Session, id: int, status: str) -> Optional[Run]:
        """Update run status."""
        return self.update(db, id, status=status)

    def complete_run(
        self,
        db: Session,
        run_id: int,
        status: str = "completed",
    ) -> Optional[Run]:
        """Mark run as completed or failed with end_time timestamp.

        Args:
            db: Database session
            run_id: Run ID to complete
            status: Final status ("completed" or "failed")

        Returns:
            Updated Run object or None if run not found
        """
        self.logger.info(f"Completing run {run_id} with status: {status}")

        run = db.query(self.model).filter(self.model.id == run_id).first()
        if not run:
            self.logger.warning(f"Run {run_id} not found - cannot complete")
            return None

        # Log current state before update
        self.logger.info(
            f"Run {run_id} current state: status={run.status}, "
            f"start_time={run.start_time}, end_time={run.end_time}",
        )

        # Update end_time and status
        updated_run = self.update(db, run_id, end_time=datetime.now(), status=status)

        # Log updated state
        if updated_run:
            self.logger.info(
                f"Run {run_id} updated: status={updated_run.status}, "
                f"end_time={updated_run.end_time}, "
                f"duration={(updated_run.end_time - updated_run.start_time).total_seconds() / 60:.2f} minutes",  # noqa: E501
            )

        return updated_run

    def get_by_status_and_mode(
        self,
        db: Session,
        status: str,
        training_mode: Optional[str] = None,
    ) -> List[Run]:
        """
        Get runs filtered by status and optionally by training_mode.

        Uses database-level filtering for efficiency.

        Args:
            db: Database session
            status: Run status ("completed", "in_progress", "failed")
            training_mode: Optional training mode filter ("centralized", "federated")

        Returns:
            List of Run objects matching criteria
        """
        query = (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.status == status)
        )

        if training_mode:
            query = query.filter(self.model.training_mode == training_mode)

        return query.order_by(self.model.start_time.desc()).all()

    def get_by_wandb_id(self, db: Session, wandb_id: str) -> Optional[Run]:
        """Get run by W&B ID."""
        return (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.wandb_id == wandb_id)
            .first()
        )

    def get_completed_runs(
        self,
        db: Session,
        experiment_id: Optional[int] = None,
    ) -> List[Run]:
        """Get all completed runs, optionally filtered by experiment."""
        query = (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.status == "completed")
        )
        if experiment_id:
            query = query.filter(self.model.experiment_id == experiment_id)
        return query.order_by(self.model.end_time.desc()).all()

    def get_failed_runs(
        self,
        db: Session,
        experiment_id: Optional[int] = None,
    ) -> List[Run]:
        """Get all failed runs, optionally filtered by experiment."""
        query = (
            db.query(self.model)
            .options(
                selectinload(Run.metrics),
                selectinload(Run.clients),
                selectinload(Run.server_evaluations),
            )
            .filter(self.model.status == "failed")
        )
        if experiment_id:
            query = query.filter(self.model.experiment_id == experiment_id)
        return query.order_by(self.model.end_time.desc()).all()

    def persist_metrics(
        self,
        db: Session,
        run_id: int,
        epoch_metrics: List[Dict[str, Any]],
        federated_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist collected metrics to database.

        Args:
            db: Database session
            run_id: Run ID to associate metrics with
            epoch_metrics: List of epoch metric dictionaries
            federated_context: Optional dict with 'client_id' and 'round_number' for federated mode  # noqa: E501
        """
        try:
            client_id, round_id = self._resolve_federated_context(db, federated_context)

            metrics_to_persist = []
            for epoch_data in epoch_metrics:
                metrics_to_persist.extend(
                    self._transform_epoch_to_metrics(
                        epoch_data,
                        run_id,
                        client_id,
                        round_id,
                    )
                )

            if metrics_to_persist:
                run_metric_crud.bulk_create(db, metrics_to_persist)
                db.commit()
                self.logger.info(
                    f"Persisted {len(metrics_to_persist)} metrics to database "
                    f"for run_id={run_id}"
                    + (
                        f", client_id={client_id}, round_id={round_id}"
                        if client_id
                        else ""
                    ),
                )

        except Exception as e:
            self.logger.error(f"Failed to persist metrics to database: {e}")
            db.rollback()
            raise

    def list_with_filters(
        self,
        db: Session,
        status: Optional[str] = None,
        training_mode: Optional[str] = None,
        sort_by: str = "start_time",
        sort_order: str = "desc",
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Run], int]:
        """
        List runs with filters, sorting, and pagination.

        Args:
            db: Database session
            status: Filter by run status (optional)
            training_mode: Filter by training mode (optional)
            sort_by: Field to sort by (default: "start_time")
            sort_order: Sort direction "asc" or "desc" (default: "desc")
            limit: Maximum number of runs to return (default: 100)
            offset: Number of runs to skip (default: 0)

        Returns:
            Tuple of (list of Run objects, total count)
        """
        query = db.query(self.model)

        if status:
            query = query.filter(self.model.status == status)
        if training_mode:
            query = query.filter(self.model.training_mode == training_mode)

        total_count = query.count()

        sort_column = getattr(self.model, sort_by, self.model.start_time)
        if sort_order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Eager load metrics and clients to prevent N+1 queries
        runs = (
            query.options(selectinload(Run.metrics), selectinload(Run.clients))
            .limit(limit)
            .offset(offset)
            .all()
        )
        return runs, total_count

    def batch_get_final_metrics(
        self,
        db: Session,
        run_ids: List[int],
    ) -> Dict[int, Dict[str, float]]:
        """
        Batch fetch final metrics for centralized runs.

        Retrieves all metrics with names starting with "final_" for the given run IDs.

        Args:
            db: Database session
            run_ids: List of run IDs to fetch metrics for

        Returns:
            Dictionary mapping run_id to dict of metric_name -> metric_value
        """
        if not run_ids:
            return {}

        final_stats_map: Dict[int, Dict[str, float]] = {}

        final_metrics = (
            db.query(RunMetric)
            .filter(
                RunMetric.run_id.in_(run_ids),
                RunMetric.metric_name.like("final_%"),
            )
            .all()
        )

        # Group by run_id and strip "final_" prefix from metric names
        for metric in final_metrics:
            if metric.run_id not in final_stats_map:
                final_stats_map[metric.run_id] = {}
            stat_name = metric.metric_name.replace("final_", "")
            final_stats_map[metric.run_id][stat_name] = metric.metric_value

        return final_stats_map

    def batch_get_federated_final_stats(
        self,
        db: Session,
        run_ids: List[int],
    ) -> Dict[int, Dict[str, float]]:
        """
        Batch fetch final stats for federated runs from server evaluations.

        Retrieves the latest server evaluation for each run and extracts
        "final_epoch_stats" from additional_metrics.

        Args:
            db: Database session
            run_ids: List of run IDs to fetch stats for

        Returns:
            Dictionary mapping run_id to final_epoch_stats dict
        """
        if not run_ids:
            return {}

        final_stats_map: Dict[int, Dict[str, float]] = {}

        # Subquery for max round per run
        max_rounds = (
            db.query(
                ServerEvaluation.run_id,
                func.max(ServerEvaluation.round_number).label("max_round"),
            )
            .filter(ServerEvaluation.run_id.in_(run_ids))
            .group_by(ServerEvaluation.run_id)
            .subquery()
        )

        # Get last evaluation for each run
        last_evals = (
            db.query(ServerEvaluation)
            .join(
                max_rounds,
                (ServerEvaluation.run_id == max_rounds.c.run_id)
                & (ServerEvaluation.round_number == max_rounds.c.max_round),
            )
            .all()
        )

        # Extract final_epoch_stats from additional_metrics
        for eval in last_evals:
            if (
                eval.additional_metrics
                and "final_epoch_stats" in eval.additional_metrics
            ):
                final_stats_map[eval.run_id] = eval.additional_metrics[
                    "final_epoch_stats"
                ]

        return final_stats_map

    def _resolve_federated_context(
        self,
        db: Session,
        federated_context: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Resolve federated context to client_id and round_id.

        Args:
            db: Database session
            federated_context: Optional dict with 'client_id' and 'round_number'

        Returns:
            Tuple of (client_id, round_id) or (None, None) if not federated
        """
        client_id = None
        round_id = None

        if federated_context:
            client_id = federated_context.get("client_id")
            round_number = federated_context.get("round_number", 0)

            if client_id is not None:
                from federated_pneumonia_detection.src.boundary.CRUD.round import (
                    round_crud,
                )

                round_id = round_crud.get_or_create_round(
                    db,
                    client_id,
                    round_number,
                )
                self.logger.info(
                    f"[persist_metrics] Federated context: "
                    f"client_id={client_id}, round_id={round_id}",
                )

        return client_id, round_id

    def _determine_dataset_type(self, metric_name: str) -> str:
        """
        Determine dataset type from metric name prefix.

        Args:
            metric_name: Name of the metric

        Returns:
            Dataset type: "train", "validation", "test", or "other"
        """
        if metric_name.startswith("train_"):
            return "train"
        elif metric_name.startswith("val_"):
            return "validation"
        elif metric_name.startswith("test_"):
            return "test"
        else:
            return "other"

    def _transform_epoch_to_metrics(
        self,
        epoch_data: Dict[str, Any],
        run_id: int,
        client_id: Optional[int],
        round_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Transform epoch data into metric dictionaries for persistence.

        Args:
            epoch_data: Dictionary containing epoch metrics
            run_id: Run ID to associate with metrics
            client_id: Optional client ID for federated runs
            round_id: Optional round ID for federated runs

        Returns:
            List of metric dictionaries ready for persistence
        """
        metrics_list = []
        epoch = epoch_data.get("epoch", 0)

        for key, value in epoch_data.items():
            if key in ["epoch", "timestamp", "global_step"]:
                continue

            if not isinstance(value, (int, float)):
                continue

            dataset_type = self._determine_dataset_type(key)

            metric_dict = {
                "run_id": run_id,
                "metric_name": key,
                "metric_value": float(value),
                "step": epoch,
                "dataset_type": dataset_type,
            }

            if client_id is not None:
                metric_dict["client_id"] = client_id
            if round_id is not None:
                metric_dict["round_id"] = round_id

            metrics_list.append(metric_dict)

        return metrics_list


run_crud = RunCRUD()
