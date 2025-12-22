"""
Centralized experiment runner for comparative analysis.

Wraps CentralizedTrainer to execute multiple training runs
with different seeds for statistical analysis.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.config import AnalysisConfig
from analysis.schemas.experiment import ExperimentResult
from analysis.modeling.seed_manager import SeedManager

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
    CentralizedTrainer,
)


class CentralizedExperimentRunner:
    """
    Executes multiple centralized training runs for comparison.

    Wraps CentralizedTrainer to:
    - Run N experiments with different seeds
    - Collect metrics from each run
    - Prepare results for statistical analysis
    """

    def __init__(
        self,
        analysis_config: AnalysisConfig,
        seed_manager: SeedManager,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize runner.

        Args:
            analysis_config: Analysis configuration
            seed_manager: SeedManager with generated seeds
            logger: Optional logger instance
        """
        self.config = analysis_config
        self.seed_manager = seed_manager
        self.logger = logger or logging.getLogger(__name__)
        self._results: List[ExperimentResult] = []

    def run_all(self) -> List[ExperimentResult]:
        """
        Execute all centralized training runs.

        Returns:
            List of ExperimentResult for each run
        """
        self._results = []
        n_runs = self.config.experiment.n_runs

        self.logger.info(f"Starting {n_runs} centralized training runs")

        for run_idx in range(n_runs):
            seed = self.seed_manager.get_seed(run_idx)
            self.logger.info(f"Run {run_idx + 1}/{n_runs} with seed {seed}")

            try:
                result = self._run_single(run_idx, seed)
                self._results.append(result)
                self.logger.info(
                    f"Run {run_idx + 1} complete: "
                    f"accuracy={result.final_metrics.get('accuracy', 0):.4f}"
                )
            except Exception as e:
                self.logger.error(f"CentralizedExperimentRunner:run_all - {type(e).__name__}: {e}")
                raise

        self.logger.info(f"Completed {len(self._results)} centralized runs")
        return self._results

    def _run_single(self, run_idx: int, seed: int) -> ExperimentResult:
        """
        Execute a single training run.

        Args:
            run_idx: Run index (0-based)
            seed: Random seed for this run

        Returns:
            ExperimentResult for this run
        """
        run_id = run_idx + 1
        experiment_name = f"centralized_run_{run_id}"

        checkpoint_dir = str(
            self.config.output.output_dir / "experiments" / "centralized" / f"run_{run_id}" / "checkpoints"
        )
        logs_dir = str(
            self.config.output.output_dir / "experiments" / "centralized" / f"run_{run_id}" / "logs"
        )

        config_manager = self._create_config_with_seed(seed)

        trainer = CentralizedTrainer(
            config_path=None,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir,
        )
        trainer.config = config_manager

        start_time = time.time()

        try:
            results = trainer.train(
                source_path=str(self.config.data.source_path),
                experiment_name=experiment_name,
                csv_filename=self.config.data.csv_filename,
                run_id=run_id,
            )
        except Exception as e:
            self.logger.error(f"CentralizedExperimentRunner:_run_single - {type(e).__name__}: {e}")
            raise

        training_duration = time.time() - start_time

        final_metrics = self._extract_final_metrics(results)
        metrics_history = self._clean_metrics_history(results.get("metrics_history", []))
        best_epoch = self._find_best_epoch(metrics_history)

        return ExperimentResult(
            run_id=run_id,
            training_mode="centralized",
            seed=seed,
            final_metrics=final_metrics,
            metrics_history=metrics_history,
            training_duration_seconds=training_duration,
            best_epoch=best_epoch,
            best_model_path=results.get("best_model_path"),
            config_snapshot=self._get_config_snapshot(config_manager),
        )

    def _create_config_with_seed(self, seed: int) -> ConfigManager:
        """Create ConfigManager with updated seed."""
        config = ConfigManager()

        config.set("experiment.seed", seed)
        config.set("experiment.epochs", self.config.experiment.epochs)
        config.set("experiment.batch_size", self.config.experiment.batch_size)
        config.set("experiment.learning_rate", self.config.experiment.learning_rate)

        return config

    def _clean_metrics_history(self, history: List[Dict]) -> List[Dict[str, float]]:
        """Clean metrics history to only include float values.

        Filters out non-numeric fields like 'timestamp' that would fail validation.
        """
        cleaned = []
        for epoch in history:
            cleaned_epoch = {}
            for key, value in epoch.items():
                if isinstance(value, (int, float)):
                    cleaned_epoch[key] = float(value)
            cleaned.append(cleaned_epoch)
        return cleaned

    def _extract_final_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract final epoch metrics from training results."""
        history = results.get("metrics_history", [])
        if not history:
            return {}

        final_epoch = history[-1]

        return {
            "accuracy": final_epoch.get("val_acc", 0.0),
            "precision": final_epoch.get("val_precision", 0.0),
            "recall": final_epoch.get("val_recall", 0.0),
            "f1": final_epoch.get("val_f1", 0.0),
            "auroc": final_epoch.get("val_auroc", 0.0),
            "loss": final_epoch.get("val_loss", 0.0),
        }

    def _find_best_epoch(self, history: List[Dict]) -> int:
        """Find epoch with best validation recall."""
        if not history:
            return 0

        best_idx = 0
        best_recall = 0.0

        for idx, epoch in enumerate(history):
            recall = epoch.get("val_recall", 0.0)
            if recall > best_recall:
                best_recall = recall
                best_idx = idx

        return best_idx + 1

    def _get_config_snapshot(self, config: ConfigManager) -> Dict[str, Any]:
        """Get configuration snapshot for documentation."""
        return {
            "epochs": config.get("experiment.epochs"),
            "batch_size": config.get("experiment.batch_size"),
            "learning_rate": config.get("experiment.learning_rate"),
            "seed": config.get("experiment.seed"),
        }

    @property
    def results(self) -> List[ExperimentResult]:
        """Get collected results."""
        return self._results
