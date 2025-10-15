"""
Results analyzer and visualization for pneumonia detection experiments.
Analyzes training logs, metrics, and generates comparison visualizations.

Usage:
    python analyze_results.py --experiment-dir experiment_results/experiment_20250115_123456
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class ResultsAnalyzer:
    """
    Analyzes and visualizes experiment results.
    Generates reports comparing centralized and federated approaches.
    """

    def __init__(self, experiment_dir: str):
        """
        Initialize results analyzer.

        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.logger = self._setup_logging()

        if not self.experiment_dir.exists():
            raise ValueError(f"Experiment directory does not exist: {experiment_dir}")

        self.logger.info(f"Analyzing results from: {self.experiment_dir}")

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze all results in the experiment directory.

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'experiment_dir': str(self.experiment_dir),
            'centralized': None,
            'federated': None,
            'comparison': None
        }

        # Analyze centralized results
        cent_dir = self.experiment_dir / "centralized"
        if cent_dir.exists():
            self.logger.info("\nAnalyzing centralized results...")
            analysis['centralized'] = self._analyze_centralized(cent_dir)

        # Analyze federated results
        fed_dir = self.experiment_dir / "federated"
        if fed_dir.exists():
            self.logger.info("\nAnalyzing federated results...")
            analysis['federated'] = self._analyze_federated(fed_dir)

        # Load comparison if exists
        comparison_file = self.experiment_dir / "comparison_report.json"
        if comparison_file.exists():
            self.logger.info("\nLoading comparison report...")
            analysis['comparison'] = self._load_json(comparison_file)

        # Generate summary
        self._print_summary(analysis)

        # Save analysis
        output_file = self.experiment_dir / "analysis_report.json"
        self._save_json(analysis, output_file)

        return analysis

    def _analyze_centralized(self, cent_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze centralized training results."""
        results_file = cent_dir / "results.json"

        if not results_file.exists():
            self.logger.warning(f"Centralized results not found: {results_file}")
            return None

        results = self._load_json(results_file)

        analysis = {
            'results_file': str(results_file),
            'checkpoint_dir': str(cent_dir / "checkpoints"),
            'logs_dir': str(cent_dir / "logs"),
            'metrics': results.get('final_metrics', {}),
            'best_model': results.get('best_model_path'),
            'training_time': results.get('training_time'),
        }

        # Log key findings
        self.logger.info("  ✓ Centralized results loaded")
        if analysis['metrics']:
            self.logger.info(f"  Metrics: {list(analysis['metrics'].keys())}")

        return analysis

    def _analyze_federated(self, fed_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze federated learning results."""
        results_file = fed_dir / "results.json"

        if not results_file.exists():
            self.logger.warning(f"Federated results not found: {results_file}")
            return None

        results = self._load_json(results_file)

        analysis = {
            'results_file': str(results_file),
            'checkpoint_dir': str(fed_dir / "checkpoints"),
            'logs_dir': str(fed_dir / "logs"),
            'num_clients': results.get('num_clients'),
            'num_rounds': results.get('num_rounds'),
            'partition_strategy': results.get('partition_strategy'),
            'status': results.get('status'),
            'metrics': results.get('metrics', {})
        }

        # Extract final metrics
        if 'losses_distributed' in analysis['metrics']:
            losses = analysis['metrics']['losses_distributed']
            if losses:
                analysis['final_loss'] = losses[-1][1] if isinstance(losses[-1], (list, tuple)) else losses[-1]
                analysis['loss_history'] = [l[1] if isinstance(l, (list, tuple)) else l for l in losses]

        # Log key findings
        self.logger.info("  ✓ Federated results loaded")
        self.logger.info(f"  Clients: {analysis['num_clients']}, Rounds: {analysis['num_rounds']}")
        if 'final_loss' in analysis:
            self.logger.info(f"  Final loss: {analysis['final_loss']:.4f}")

        return analysis

    def _print_summary(self, analysis: Dict[str, Any]) -> None:
        """Print summary of analysis."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANALYSIS SUMMARY")
        self.logger.info("=" * 80)

        # Centralized summary
        if analysis['centralized']:
            self.logger.info("\nCentralized Training:")
            cent = analysis['centralized']
            self.logger.info(f"  Best model: {cent.get('best_model', 'N/A')}")
            if cent.get('metrics'):
                self.logger.info("  Metrics:")
                for key, value in cent['metrics'].items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"    {key}: {value:.4f}")

        # Federated summary
        if analysis['federated']:
            self.logger.info("\nFederated Learning:")
            fed = analysis['federated']
            self.logger.info(f"  Status: {fed.get('status', 'N/A')}")
            self.logger.info(f"  Clients: {fed.get('num_clients', 'N/A')}")
            self.logger.info(f"  Rounds: {fed.get('num_rounds', 'N/A')}")
            self.logger.info(f"  Strategy: {fed.get('partition_strategy', 'N/A')}")
            if 'final_loss' in fed:
                self.logger.info(f"  Final loss: {fed['final_loss']:.4f}")

        # Comparison
        if analysis['comparison']:
            self.logger.info("\nComparison Available: Yes")
        else:
            self.logger.info("\nComparison Available: No")

        self.logger.info("\n" + "=" * 80)

    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {e}")
            return {}

    def _save_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save data to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"\n✓ Analysis saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {e}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('ResultsAnalyzer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger


def find_latest_experiment(base_dir: str = "experiment_results") -> Optional[Path]:
    """Find the most recent experiment directory."""
    base_path = Path(base_dir)

    if not base_path.exists():
        return None

    # Find all experiment directories
    exp_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("experiment_")]

    if not exp_dirs:
        return None

    # Sort by modification time and return latest
    latest = max(exp_dirs, key=lambda d: d.stat().st_mtime)
    return latest


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze pneumonia detection experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--experiment-dir',
        type=str,
        default=None,
        help='Path to experiment directory (default: latest in experiment_results/)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine experiment directory
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
    else:
        latest = find_latest_experiment()
        if latest:
            experiment_dir = str(latest)
            print(f"Using latest experiment: {experiment_dir}")
        else:
            print("Error: No experiment directory found. Specify with --experiment-dir")
            return 1

    try:
        # Run analysis
        analyzer = ResultsAnalyzer(experiment_dir)
        analysis = analyzer.analyze()

        print("\n✓ Analysis complete!")
        return 0

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
