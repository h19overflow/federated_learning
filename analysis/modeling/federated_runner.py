"""
Federated experiment runner for comparative analysis.

Invokes federated training via subprocess and collects results
for multiple runs with different seeds.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.config import AnalysisConfig
from analysis.schemas.experiment import ExperimentResult
from analysis.modeling.seed_manager import SeedManager


class FederatedExperimentRunner:
    """
    Executes multiple federated training runs for comparison.

    Invokes federated training subprocess and:
    - Runs N experiments with different seeds
    - Parses results JSON after completion
    - Extracts server evaluation metrics
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
        Execute all federated training runs.

        Returns:
            List of ExperimentResult for each run
        """
        self._results = []
        n_runs = self.config.experiment.n_runs

        self.logger.info(f"Starting {n_runs} federated training runs")

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
                self.logger.error(f"FederatedExperimentRunner:run_all - {type(e).__name__}: {e}")
                raise

        self.logger.info(f"Completed {len(self._results)} federated runs")
        return self._results

    def _run_single(self, run_idx: int, seed: int) -> ExperimentResult:
        """
        Execute a single federated training run.

        Args:
            run_idx: Run index (0-based)
            seed: Random seed for this run

        Returns:
            ExperimentResult for this run
        """
        run_id = run_idx + 1
        experiment_name = f"federated_run_{run_id}"

        output_dir = self.config.output.output_dir / "experiments" / "federated" / f"run_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        self._update_federated_config(seed)

        start_time = time.time()

        try:
            self._invoke_federated_training(
                source_path=str(self.config.data.source_path),
                csv_filename=self.config.data.csv_filename,
                num_rounds=self.config.federated.num_rounds,
                num_clients=self.config.federated.num_clients,
                seed=seed,
                run_id=run_id,
            )
        except Exception as e:
            self.logger.error(f"FederatedExperimentRunner:_run_single - {type(e).__name__}: {e}")
            raise

        training_duration = time.time() - start_time

        # Find results file using the run_id passed via env var
        results_file = self._find_results_file(run_id)
        if results_file and results_file.exists():
            raw_results = self._parse_results(results_file)
        else:
            raw_results = {}

        final_metrics = self._extract_final_metrics(raw_results)
        metrics_history = self._build_metrics_history(raw_results)
        best_epoch = self._find_best_round(metrics_history)

        return ExperimentResult(
            run_id=run_id,
            training_mode="federated",
            seed=seed,
            final_metrics=final_metrics,
            metrics_history=metrics_history,
            training_duration_seconds=training_duration,
            best_epoch=best_epoch,
            best_model_path=None,
            config_snapshot=self._get_config_snapshot(seed),
        )

    def _update_federated_config(self, seed: int) -> None:
        """Update federated config with seed and parameters."""
        from federated_pneumonia_detection.config.config_manager import ConfigManager

        config = ConfigManager()
        config.set("experiment.seed", seed)
        config.set("federated.num_clients", self.config.federated.num_clients)
        config.set("federated.num_rounds", self.config.federated.num_rounds)
        config.set("federated.local_epochs", self.config.federated.local_epochs)

    def _invoke_federated_training(
        self,
        source_path: str,
        csv_filename: str,
        num_rounds: int,
        num_clients: int,
        seed: int,
        run_id: int,
    ) -> None:
        """Invoke federated training via subprocess."""
        # Use absolute paths to avoid cwd/relative path mismatch
        script_path = Path("federated_pneumonia_detection/src/rf.ps1").resolve()
        federated_dir = Path("federated_pneumonia_detection/src/control/federated_new_version").resolve()

        cmd = [
            "powershell",
            "-ExecutionPolicy", "Bypass",
            "-File", str(script_path),
        ]

        # Pass all parameters via environment variables for subprocess isolation
        env_vars = {
            "FL_SOURCE_PATH": source_path,
            "FL_CSV_FILENAME": csv_filename,
            "FL_NUM_ROUNDS": str(num_rounds),
            "FL_NUM_CLIENTS": str(num_clients),
            "FL_SEED": str(seed),
            "FL_RUN_ID": str(run_id),
        }

        try:
            import os
            env = os.environ.copy()
            env.update(env_vars)

            result = subprocess.run(
                cmd,
                cwd=str(federated_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            # Verbose subprocess output logging for debugging
            self.logger.info(f"STDOUT: {result.stdout}")
            self.logger.info(f"STDERR: {result.stderr}")
            self.logger.info(f"Return code: {result.returncode}")

            if result.returncode != 0:
                self.logger.error(f"Federated training failed: {result.stderr}")
                raise RuntimeError(f"Federated training failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.logger.error("Federated training timed out")
            raise
        except Exception as e:
            self.logger.error(f"FederatedExperimentRunner:_invoke_federated_training - {type(e).__name__}: {e}")
            raise

    def _find_results_file(self, run_id: int) -> Optional[Path]:
        """Find results JSON file by run_id.

        Args:
            run_id: The database run_id used to name the results file

        Returns:
            Path to results file if found, None otherwise
        """
        # First try exact match by run_id (created by server_app.py)
        results_dir = Path(".")
        exact_file = results_dir / f"results_{run_id}.json"

        if exact_file.exists():
            self.logger.info(f"Found results file: {exact_file}")
            return exact_file

        # Fallback: search for most recent results file (backward compatibility)
        pattern = "results_*.json"
        files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        if files:
            self.logger.warning(
                f"Exact results file not found for run_id={run_id}, using most recent: {files[0]}"
            )
            return files[0]

        return None

    def _parse_results(self, results_file: Path) -> Dict[str, Any]:
        """Parse federated results JSON file."""
        try:
            with open(results_file) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"FederatedExperimentRunner:_parse_results - {type(e).__name__}: {e}")
            return {}

    def _extract_final_metrics(self, raw_results: Dict) -> Dict[str, float]:
        """Extract final round server evaluation metrics."""
        server_metrics = raw_results.get("evaluate_metrics_serverapp", {})

        if not server_metrics:
            return {}

        last_round_key = max(server_metrics.keys(), key=int)
        last_round = server_metrics[last_round_key]

        if isinstance(last_round, str):
            try:
                last_round = eval(last_round)
            except Exception:
                return {}

        return {
            "accuracy": last_round.get("server_accuracy", 0.0),
            "precision": last_round.get("server_precision", 0.0),
            "recall": last_round.get("server_recall", 0.0),
            "f1": last_round.get("server_f1", 0.0),
            "auroc": last_round.get("server_auroc", 0.0),
            "loss": last_round.get("server_loss", 0.0),
        }

    def _build_metrics_history(self, raw_results: Dict) -> List[Dict[str, float]]:
        """Build metrics history from server evaluations per round."""
        server_metrics = raw_results.get("evaluate_metrics_serverapp", {})
        history = []

        for round_key in sorted(server_metrics.keys(), key=int):
            round_data = server_metrics[round_key]

            if isinstance(round_data, str):
                try:
                    round_data = eval(round_data)
                except Exception:
                    continue

            history.append({
                "round": int(round_key),
                "val_acc": round_data.get("server_accuracy", 0.0),
                "val_precision": round_data.get("server_precision", 0.0),
                "val_recall": round_data.get("server_recall", 0.0),
                "val_f1": round_data.get("server_f1", 0.0),
                "val_auroc": round_data.get("server_auroc", 0.0),
                "val_loss": round_data.get("server_loss", 0.0),
            })

        return history

    def _find_best_round(self, history: List[Dict]) -> int:
        """Find round with best server recall."""
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

    def _get_config_snapshot(self, seed: int) -> Dict[str, Any]:
        """Get configuration snapshot for documentation."""
        return {
            "num_clients": self.config.federated.num_clients,
            "num_rounds": self.config.federated.num_rounds,
            "local_epochs": self.config.federated.local_epochs,
            "seed": seed,
        }

    @property
    def results(self) -> List[ExperimentResult]:
        """Get collected results."""
        return self._results
