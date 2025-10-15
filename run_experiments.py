"""
Comprehensive experiment runner for pneumonia detection.
Runs centralized training, federated learning, and comparison experiments with full logging.

Usage:
    # Run complete comparison
    python run_experiments.py --data-path data.zip --mode comparison

    # Run only centralized
    python run_experiments.py --data-path data.zip --mode centralized

    # Run only federated
    python run_experiments.py --data-path data.zip --mode federated --partition-strategy iid

    # Custom configuration
    python run_experiments.py --data-path data.zip --config config/custom.yaml --output-dir my_results
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer
from federated_pneumonia_detection.src.control.federated_learning.federated_trainer import FederatedTrainer
from federated_pneumonia_detection.src.control.comparison.experiment_orchestrator import ExperimentOrchestrator


class ExperimentRunner:
    """
    Orchestrates complete training experiments with comprehensive logging.
    Handles centralized, federated, and comparison experiments.
    """

    def __init__(
        self,
        data_path: str,
        output_dir: str = "experiment_results",
        config_path: Optional[str] = None,
        partition_strategy: str = "iid",
        log_level: str = "INFO"
    ):
        """
        Initialize experiment runner.

        Args:
            data_path: Path to dataset (zip file or directory)
            output_dir: Directory for all outputs
            config_path: Optional path to config file
            partition_strategy: Partitioning strategy for federated learning
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.data_path = data_path
        self.config_path = config_path
        self.partition_strategy = partition_strategy

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"experiment_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging(log_level)

        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT RUNNER INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"Data path: {data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Config path: {config_path or 'Default'}")
        self.logger.info(f"Partition strategy: {partition_strategy}")
        self.logger.info("=" * 80 + "\n")

    def run_centralized(self, experiment_name: str = "centralized_training") -> Dict[str, Any]:
        """
        Run centralized training experiment.

        Args:
            experiment_name: Name for this experiment

        Returns:
            Dictionary with training results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING CENTRALIZED TRAINING")
        self.logger.info("=" * 80 + "\n")

        # Create centralized output directory
        centralized_dir = self.output_dir / "centralized"
        checkpoint_dir = centralized_dir / "checkpoints"
        logs_dir = centralized_dir / "logs"

        try:
            # Initialize trainer
            self.logger.info("Initializing CentralizedTrainer...")
            trainer = CentralizedTrainer(
                config_path=self.config_path,
                checkpoint_dir=str(checkpoint_dir),
                logs_dir=str(logs_dir)
            )

            # Validate data
            self.logger.info("Validating data source...")
            validation = trainer.validate_source(self.data_path)
            if not validation.get('valid', False):
                raise ValueError(f"Data validation failed: {validation.get('error')}")
            self.logger.info("✓ Data validation passed")

            # Run training
            self.logger.info(f"\nStarting training from: {self.data_path}")
            results = trainer.train(
                source_path=self.data_path,
                experiment_name=experiment_name
            )

            # Save results
            results_file = centralized_dir / "results.json"
            self._save_results(results, results_file)

            # Log summary
            self._log_centralized_summary(results)

            return {
                'status': 'success',
                'results': results,
                'output_dir': str(centralized_dir),
                'checkpoint_dir': str(checkpoint_dir),
                'logs_dir': str(logs_dir)
            }

        except Exception as e:
            self.logger.error(f"\n✗ CENTRALIZED TRAINING FAILED: {e}")
            self.logger.exception("Full traceback:")
            return {
                'status': 'failed',
                'error': str(e),
                'output_dir': str(centralized_dir)
            }

    def run_federated(self, experiment_name: str = "federated_training") -> Dict[str, Any]:
        """
        Run federated learning experiment.

        Args:
            experiment_name: Name for this experiment

        Returns:
            Dictionary with training results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING FEDERATED LEARNING")
        self.logger.info("=" * 80 + "\n")

        # Create federated output directory
        federated_dir = self.output_dir / "federated"
        checkpoint_dir = federated_dir / "checkpoints"
        logs_dir = federated_dir / "logs"

        try:
            # Initialize trainer
            self.logger.info("Initializing FederatedTrainer...")
            self.logger.info(f"Partition strategy: {self.partition_strategy}")
            trainer = FederatedTrainer(
                config_path=self.config_path,
                checkpoint_dir=str(checkpoint_dir),
                logs_dir=str(logs_dir),
                partition_strategy=self.partition_strategy
            )

            # Validate data
            self.logger.info("Validating data source...")
            validation = trainer.validate_source(self.data_path)
            if not validation.get('valid', False):
                raise ValueError(f"Data validation failed: {validation.get('error')}")
            self.logger.info("✓ Data validation passed")

            # Run training
            self.logger.info(f"\nStarting federated training from: {self.data_path}")
            results = trainer.train(
                source_path=self.data_path,
                experiment_name=experiment_name
            )

            # Save results
            results_file = federated_dir / "results.json"
            self._save_results(results, results_file)

            # Log summary
            self._log_federated_summary(results)

            return {
                'status': 'success',
                'results': results,
                'output_dir': str(federated_dir),
                'checkpoint_dir': str(checkpoint_dir),
                'logs_dir': str(logs_dir),
                'partition_strategy': self.partition_strategy
            }

        except Exception as e:
            self.logger.error(f"\n✗ FEDERATED TRAINING FAILED: {e}")
            self.logger.exception("Full traceback:")
            return {
                'status': 'failed',
                'error': str(e),
                'output_dir': str(federated_dir)
            }

    def run_comparison(
        self,
        centralized_name: str = "centralized_training",
        federated_name: str = "federated_training"
    ) -> Dict[str, Any]:
        """
        Run both centralized and federated training, then compare.

        Args:
            centralized_name: Name for centralized experiment
            federated_name: Name for federated experiment

        Returns:
            Dictionary with comparison results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING COMPARISON EXPERIMENT")
        self.logger.info("=" * 80 + "\n")

        try:
            # Use ExperimentOrchestrator for comparison
            orchestrator = ExperimentOrchestrator(
                config_path=self.config_path,
                base_output_dir=str(self.output_dir.parent),
                partition_strategy=self.partition_strategy
            )

            # Run comparison
            comparison = orchestrator.run_comparison(
                source_path=self.data_path,
                centralized_name=centralized_name,
                federated_name=federated_name
            )

            # Save comparison results
            comparison_file = Path(orchestrator.experiment_dir) / "comparison_summary.json"
            self._save_results(comparison, comparison_file)

            # Generate detailed report
            self._generate_comparison_report(comparison)

            return comparison

        except Exception as e:
            self.logger.error(f"\n✗ COMPARISON EXPERIMENT FAILED: {e}")
            self.logger.exception("Full traceback:")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _log_centralized_summary(self, results: Dict[str, Any]) -> None:
        """Log summary of centralized training results."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CENTRALIZED TRAINING SUMMARY")
        self.logger.info("=" * 80)

        if 'best_model_path' in results:
            self.logger.info(f"✓ Best model: {results['best_model_path']}")

        if 'final_metrics' in results:
            metrics = results['final_metrics']
            self.logger.info("\nFinal Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 80 + "\n")

    def _log_federated_summary(self, results: Dict[str, Any]) -> None:
        """Log summary of federated training results."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FEDERATED LEARNING SUMMARY")
        self.logger.info("=" * 80)

        self.logger.info(f"Status: {results.get('status', 'unknown')}")
        self.logger.info(f"Number of clients: {results.get('num_clients', 'N/A')}")
        self.logger.info(f"Number of rounds: {results.get('num_rounds', 'N/A')}")
        self.logger.info(f"Partition strategy: {results.get('partition_strategy', 'N/A')}")

        if 'metrics' in results:
            metrics = results['metrics']
            if 'losses_distributed' in metrics and metrics['losses_distributed']:
                final_loss = metrics['losses_distributed'][-1][1]
                self.logger.info(f"\nFinal distributed loss: {final_loss:.4f}")

        self.logger.info("=" * 80 + "\n")

    def _generate_comparison_report(self, comparison: Dict[str, Any]) -> None:
        """Generate detailed comparison report."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPARISON REPORT")
        self.logger.info("=" * 80)

        cent_status = comparison.get('centralized', {}).get('status', 'unknown')
        fed_status = comparison.get('federated', {}).get('status', 'unknown')

        self.logger.info(f"\nCentralized Training: {cent_status.upper()}")
        self.logger.info(f"Federated Learning: {fed_status.upper()}")

        if cent_status == 'success' and fed_status == 'success':
            self.logger.info("\n✓ Both training methods completed successfully!")

            # Compare metrics if available
            comp_metrics = comparison.get('comparison_metrics', {})
            if comp_metrics:
                self.logger.info("\nMetrics Comparison:")
                self.logger.info(json.dumps(comp_metrics, indent=2, default=str))

        self.logger.info(f"\nResults saved to: {comparison.get('experiment_dir', 'N/A')}")
        self.logger.info("=" * 80 + "\n")

    def _save_results(self, results: Dict[str, Any], filepath: Path) -> None:
        """Save results to JSON file."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"✓ Results saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"✗ Failed to save results: {e}")

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logging to both file and console."""
        logger = logging.getLogger('ExperimentRunner')
        logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        log_file = self.output_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pneumonia detection experiments with centralized and federated learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete comparison
  python run_experiments.py --data-path data.zip --mode comparison

  # Run only centralized training
  python run_experiments.py --data-path data.zip --mode centralized

  # Run only federated learning with stratified partitioning
  python run_experiments.py --data-path data.zip --mode federated --partition-strategy stratified

  # Custom configuration
  python run_experiments.py --data-path data.zip --config config/custom.yaml --output-dir my_results
        """
    )

    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to dataset (zip file or directory)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['centralized', 'federated', 'comparison'],
        default='comparison',
        help='Experiment mode: centralized, federated, or comparison (default: comparison)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration file (optional)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiment_results',
        help='Base directory for experiment outputs (default: experiment_results)'
    )

    parser.add_argument(
        '--partition-strategy',
        type=str,
        choices=['iid', 'non-iid', 'stratified'],
        default='iid',
        help='Data partitioning strategy for federated learning (default: iid)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Custom experiment name (optional)'
    )

    return parser.parse_args()


def main():
    """Main entry point for experiment runner."""
    args = parse_args()

    # Validate data path
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        return 1

    # Initialize runner
    runner = ExperimentRunner(
        data_path=args.data_path,
        output_dir=args.output_dir,
        config_path=args.config,
        partition_strategy=args.partition_strategy,
        log_level=args.log_level
    )

    # Run experiment based on mode
    try:
        if args.mode == 'centralized':
            exp_name = args.experiment_name or 'centralized_training'
            results = runner.run_centralized(exp_name)

        elif args.mode == 'federated':
            exp_name = args.experiment_name or 'federated_training'
            results = runner.run_federated(exp_name)

        elif args.mode == 'comparison':
            cent_name = f"{args.experiment_name}_centralized" if args.experiment_name else "centralized_training"
            fed_name = f"{args.experiment_name}_federated" if args.experiment_name else "federated_training"
            results = runner.run_comparison(cent_name, fed_name)

        # Final summary
        runner.logger.info("\n" + "=" * 80)
        runner.logger.info("EXPERIMENT COMPLETED")
        runner.logger.info("=" * 80)
        runner.logger.info(f"Mode: {args.mode}")
        runner.logger.info(f"Output directory: {runner.output_dir}")
        runner.logger.info(f"Log file: {runner.output_dir / 'experiment.log'}")

        if isinstance(results, dict) and results.get('status') == 'failed':
            runner.logger.error(f"Status: FAILED - {results.get('error')}")
            runner.logger.info("=" * 80 + "\n")
            return 1
        else:
            runner.logger.info("Status: SUCCESS")
            runner.logger.info("=" * 80 + "\n")
            return 0

    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
