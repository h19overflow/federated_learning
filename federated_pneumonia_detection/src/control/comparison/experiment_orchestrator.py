"""
Experiment orchestrator for comparing centralized and federated learning approaches.
Provides unified interface for running and comparing both training methods.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer
from federated_pneumonia_detection.src.control.federated_learning.federated_trainer import FederatedTrainer


class ExperimentOrchestrator:
    """
    Orchestrates experiments comparing centralized and federated learning.

    Provides unified API for running both approaches and generating comparison reports.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        base_output_dir: str = "experiments",
        partition_strategy: str = "iid"
    ):
        """
        Initialize experiment orchestrator.

        Args:
            config_path: Path to configuration file
            base_output_dir: Base directory for all experiment outputs
            partition_strategy: Data partitioning strategy for federated learning
        """
        self.config_path = config_path
        self.base_output_dir = base_output_dir
        self.partition_strategy = partition_strategy
        self.logger = self._setup_logging()

        # Create timestamp for this experiment run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(base_output_dir, f"experiment_{self.timestamp}")

        # Create output directories
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.logger.info("ExperimentOrchestrator initialized")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")

    def run_centralized(
        self,
        source_path: str,
        experiment_name: str = "centralized_training"
    ) -> Dict[str, Any]:
        """
        Run centralized training only.

        Args:
            source_path: Path to zip file or directory containing dataset
            experiment_name: Name for this experiment

        Returns:
            Dictionary with centralized training results
        """
        self.logger.info("=" * 80)
        self.logger.info("RUNNING CENTRALIZED TRAINING")
        self.logger.info("=" * 80)

        # Create centralized trainer
        centralized_dir = os.path.join(self.experiment_dir, "centralized")
        checkpoint_dir = os.path.join(centralized_dir, "checkpoints")
        logs_dir = os.path.join(centralized_dir, "logs")

        trainer = CentralizedTrainer(
            config_path=self.config_path,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir
        )

        # Run training
        try:
            results = trainer.train(source_path, experiment_name)

            # Save results
            self._save_results(results, "centralized_results.json")

            self.logger.info("Centralized training completed successfully")
            self.logger.info(f"Results saved to: {self.experiment_dir}")

            return {
                'status': 'success',
                'results': results,
                'output_dir': centralized_dir
            }

        except Exception as e:
            self.logger.error(f"Centralized training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'output_dir': centralized_dir
            }

    def run_federated(
        self,
        source_path: str,
        experiment_name: str = "federated_training"
    ) -> Dict[str, Any]:
        """
        Run federated learning only.

        Args:
            source_path: Path to zip file or directory containing dataset
            experiment_name: Name for this experiment

        Returns:
            Dictionary with federated training results
        """
        self.logger.info("=" * 80)
        self.logger.info("RUNNING FEDERATED LEARNING")
        self.logger.info("=" * 80)

        # Create federated trainer
        federated_dir = os.path.join(self.experiment_dir, "federated")
        checkpoint_dir = os.path.join(federated_dir, "checkpoints")
        logs_dir = os.path.join(federated_dir, "logs")

        trainer = FederatedTrainer(
            config_path=self.config_path,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir,
            partition_strategy=self.partition_strategy
        )

        # Run training
        try:
            results = trainer.train(source_path, experiment_name)

            # Save results
            self._save_results(results, "federated_results.json")

            self.logger.info("Federated training completed successfully")
            self.logger.info(f"Results saved to: {self.experiment_dir}")

            return {
                'status': 'success',
                'results': results,
                'output_dir': federated_dir
            }

        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'output_dir': federated_dir
            }

    def run_comparison(
        self,
        source_path: str,
        centralized_name: str = "centralized_training",
        federated_name: str = "federated_training"
    ) -> Dict[str, Any]:
        """
        Run both centralized and federated training, then compare results.

        Args:
            source_path: Path to zip file or directory containing dataset
            centralized_name: Name for centralized experiment
            federated_name: Name for federated experiment

        Returns:
            Dictionary with comparison results
        """
        self.logger.info("=" * 80)
        self.logger.info("RUNNING COMPARISON EXPERIMENT")
        self.logger.info("=" * 80)

        # Run centralized training
        centralized_results = self.run_centralized(source_path, centralized_name)

        # Run federated training
        federated_results = self.run_federated(source_path, federated_name)

        # Generate comparison
        comparison = self._generate_comparison(centralized_results, federated_results)

        # Save comparison report
        self._save_results(comparison, "comparison_report.json")

        self.logger.info("=" * 80)
        self.logger.info("COMPARISON COMPLETED")
        self.logger.info("=" * 80)

        self._print_comparison_summary(comparison)

        return comparison

    def _generate_comparison(
        self,
        centralized_results: Dict[str, Any],
        federated_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comparison report between centralized and federated results.

        Args:
            centralized_results: Results from centralized training
            federated_results: Results from federated training

        Returns:
            Comparison report dictionary
        """
        comparison = {
            'timestamp': self.timestamp,
            'experiment_dir': self.experiment_dir,
            'centralized': centralized_results,
            'federated': federated_results,
            'comparison_metrics': {}
        }

        # Extract and compare key metrics
        if centralized_results['status'] == 'success' and federated_results['status'] == 'success':
            cent_res = centralized_results.get('results', {})
            fed_res = federated_results.get('results', {})

            # Compare final metrics if available
            cent_metrics = cent_res.get('final_metrics', {})
            fed_metrics = fed_res.get('final_metrics', {})

            comparison['comparison_metrics'] = {
                'centralized_metrics': cent_metrics,
                'federated_metrics': fed_metrics,
                'note': 'Detailed metric comparison available when training completes'
            }

        return comparison

    def _print_comparison_summary(self, comparison: Dict[str, Any]) -> None:
        """Print comparison summary to console."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPARISON SUMMARY")
        self.logger.info("=" * 80)

        cent_status = comparison['centralized']['status']
        fed_status = comparison['federated']['status']

        self.logger.info(f"Centralized Status: {cent_status}")
        self.logger.info(f"Federated Status: {fed_status}")

        if cent_status == 'success' and fed_status == 'success':
            self.logger.info("\nBoth training methods completed successfully!")
            self.logger.info(f"Results directory: {self.experiment_dir}")
        else:
            self.logger.warning("\nSome training methods failed. Check logs for details.")

        self.logger.info("=" * 80 + "\n")

    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to JSON file."""
        output_path = os.path.join(self.experiment_dir, filename)

        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def get_experiment_info(self) -> Dict[str, Any]:
        """Get information about current experiment setup."""
        return {
            'timestamp': self.timestamp,
            'experiment_dir': self.experiment_dir,
            'config_path': self.config_path,
            'partition_strategy': self.partition_strategy,
            'base_output_dir': self.base_output_dir
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger(__name__)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger


# Convenience function for quick experiments
def run_quick_comparison(
    source_path: str,
    config_path: Optional[str] = None,
    partition_strategy: str = "iid"
) -> Dict[str, Any]:
    """
    Quick convenience function to run a complete comparison experiment.

    Args:
        source_path: Path to dataset
        config_path: Optional config file path
        partition_strategy: Partitioning strategy for federated learning

    Returns:
        Comparison results
    """
    orchestrator = ExperimentOrchestrator(
        config_path=config_path,
        partition_strategy=partition_strategy
    )

    return orchestrator.run_comparison(source_path)
