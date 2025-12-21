"""
CLI entry point for comparative analysis.

Orchestrates the complete analysis pipeline from data loading
through experiment execution, statistical analysis, and report generation.

Usage:
    python -m analysis.run_analysis --source-path data/Training.zip
    python -m analysis.run_analysis --source-path data/Training.zip --n-runs 5 --output-dir results
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from analysis.config import AnalysisConfig, DataConfig, ExperimentConfig, FederatedConfig, OutputConfig
from analysis.preprocessing.data_loader import AnalysisDataLoader
from analysis.eda.data_profiler import DataProfiler
from analysis.eda.visualizer import EDAVisualizer
from analysis.modeling.seed_manager import SeedManager
from analysis.modeling.centralized_runner import CentralizedExperimentRunner
from analysis.modeling.federated_runner import FederatedExperimentRunner
from analysis.modeling.results_aggregator import ResultsAggregator
from analysis.statistics.tests import StatisticalTests
from analysis.statistics.effect_size import EffectSizeCalculator
from analysis.statistics.bootstrap import BootstrapAnalyzer
from analysis.schemas.statistics import MetricStatisticalResult, StatisticalAnalysisResult
from analysis.visualization.learning_curves import LearningCurvePlotter
from analysis.visualization.metric_bars import MetricBarPlotter
from analysis.visualization.distributions import DistributionPlotter
from analysis.visualization.roc_curves import ROCCurvePlotter
from analysis.reporting.latex_tables import LatexTableGenerator
from analysis.reporting.markdown_report import MarkdownReportGenerator
from analysis.reporting.exporters import ResultsExporter


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the analysis pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("analysis")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comparative Analysis: Federated vs Centralized Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m analysis.run_analysis --source-path data/Training.zip
    python -m analysis.run_analysis --source-path data/ --n-runs 5 --output-dir results/analysis
    python -m analysis.run_analysis --source-path data.zip --skip-training --results-dir existing_results
        """,
    )

    parser.add_argument(
        "--source-path",
        type=str,
        required=True,
        help="Path to ZIP file or directory containing dataset",
    )
    parser.add_argument(
        "--csv-filename",
        type=str,
        default="stage2_train_metadata.csv",
        help="Metadata CSV filename (default: stage2_train_metadata.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Output directory for results (default: analysis_output)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of runs per approach (default: 5)",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=42,
        help="Master seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for centralized (default: 10)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of federated clients (default: 2)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of federated rounds (default: 5)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, use existing results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory with existing results (requires --skip-training)",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA generation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def run_eda(
    config: AnalysisConfig,
    data_loader: AnalysisDataLoader,
    logger: logging.Logger,
) -> dict:
    """Run EDA pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Exploratory Data Analysis")
    logger.info("=" * 60)

    eda_dir = config.get_output_subdir("eda")

    profiler = DataProfiler(data_loader, logger)
    profile = profiler.generate_profile()
    profiler.save_profile(eda_dir / "data_profile.json")

    visualizer = EDAVisualizer(data_loader, eda_dir, config.output.figure_dpi, logger)
    visualizer.generate_all()

    logger.info("EDA complete")
    return profile


def run_experiments(
    config: AnalysisConfig,
    seed_manager: SeedManager,
    logger: logging.Logger,
):
    """Run centralized and federated experiments."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Experiment Execution")
    logger.info("=" * 60)

    logger.info(f"Running {config.experiment.n_runs} centralized experiments...")
    cent_runner = CentralizedExperimentRunner(config, seed_manager, logger)
    centralized_results = cent_runner.run_all()

    logger.info(f"Running {config.experiment.n_runs} federated experiments...")
    fed_runner = FederatedExperimentRunner(config, seed_manager, logger)
    federated_results = fed_runner.run_all()

    aggregator = ResultsAggregator(logger)
    comparative_result = aggregator.aggregate(centralized_results, federated_results)

    aggregator.save_results(config.output.output_dir / "comparative_result.json")

    logger.info("Experiments complete")
    return comparative_result


def run_statistical_analysis(
    comparative_result,
    config: AnalysisConfig,
    logger: logging.Logger,
):
    """Run statistical analysis on results."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Statistical Analysis")
    logger.info("=" * 60)

    aggregator = ResultsAggregator(logger)
    aggregator._comparative_result = comparative_result
    metrics_data = aggregator.get_metrics_for_comparison()

    tests = StatisticalTests(config.statistical.alpha, logger)
    effect_calc = EffectSizeCalculator(config.statistical.confidence_level, logger)
    bootstrap = BootstrapAnalyzer(config.statistical.n_bootstrap, config.statistical.confidence_level, logger=logger)

    metric_results = {}

    for metric_name, data in metrics_data.items():
        cent = data["centralized"]
        fed = data["federated"]

        norm_cent = tests.shapiro_wilk(cent, "centralized")
        norm_fed = tests.shapiro_wilk(fed, "federated")

        comparison_test, _ = tests.select_appropriate_test(cent, fed, metric_name)
        effect_size = effect_calc.cohens_d(cent, fed)
        bootstrap_diff = bootstrap.difference_ci(cent, fed)
        anova = tests.one_way_anova(cent, fed, metric_name)

        metric_results[metric_name] = MetricStatisticalResult(
            metric_name=metric_name,
            normality_centralized=norm_cent,
            normality_federated=norm_fed,
            comparison_test=comparison_test,
            effect_size=effect_size,
            bootstrap_difference=bootstrap_diff,
            anova=anova,
        )

    significant = [m for m, r in metric_results.items() if r.comparison_test.is_significant]

    if not significant:
        conclusion = (
            "No statistically significant differences were found between "
            "centralized and federated learning across all metrics. "
            "Federated learning achieves comparable performance."
        )
    else:
        conclusion = (
            f"Statistically significant differences found in: {', '.join(significant)}. "
            "Further analysis recommended to assess practical significance."
        )

    statistical_result = StatisticalAnalysisResult(
        metrics=metric_results,
        alpha=config.statistical.alpha,
        confidence_level=config.statistical.confidence_level,
        n_bootstrap=config.statistical.n_bootstrap,
        overall_conclusion=conclusion,
    )

    logger.info("Statistical analysis complete")
    return statistical_result


def run_visualization(
    comparative_result,
    statistical_result,
    config: AnalysisConfig,
    logger: logging.Logger,
):
    """Generate all visualizations."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Visualization")
    logger.info("=" * 60)

    figures_dir = config.get_output_subdir("figures")

    lc_plotter = LearningCurvePlotter(figures_dir, config.output.figure_dpi, logger)
    lc_plotter.generate_all(
        comparative_result.centralized_runs,
        comparative_result.federated_runs,
    )

    bar_plotter = MetricBarPlotter(figures_dir, config.output.figure_dpi, logger)
    bar_plotter.plot_comparison(comparative_result, statistical_result)
    bar_plotter.plot_difference_chart(comparative_result)

    dist_plotter = DistributionPlotter(figures_dir, config.output.figure_dpi, logger)
    dist_plotter.plot_boxplots(comparative_result)

    roc_plotter = ROCCurvePlotter(figures_dir, config.output.figure_dpi, logger)
    roc_plotter.generate_all(comparative_result)

    logger.info("Visualization complete")
    return figures_dir


def run_reporting(
    comparative_result,
    statistical_result,
    data_profile,
    figures_dir,
    config: AnalysisConfig,
    logger: logging.Logger,
):
    """Generate all reports."""
    logger.info("=" * 60)
    logger.info("PHASE 5: Report Generation")
    logger.info("=" * 60)

    tables_dir = config.get_output_subdir("tables")
    latex_gen = LatexTableGenerator(tables_dir, logger)
    latex_gen.generate_all(comparative_result, statistical_result)

    md_gen = MarkdownReportGenerator(config.output.output_dir, logger)
    md_gen.generate_report(comparative_result, statistical_result, data_profile, figures_dir)

    exporter = ResultsExporter(config.output.output_dir, logger)
    exporter.create_output_package(
        comparative_result,
        statistical_result,
        data_profile,
        [figures_dir],
    )

    logger.info("Report generation complete")


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("Comparative Analysis: Federated vs Centralized Learning")
    logger.info("=" * 60)

    config = AnalysisConfig(
        data=DataConfig(
            source_path=args.source_path,
            csv_filename=args.csv_filename,
        ),
        experiment=ExperimentConfig(
            n_runs=args.n_runs,
            master_seed=args.master_seed,
            epochs=args.epochs,
        ),
        federated=FederatedConfig(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
        ),
        output=OutputConfig(
            output_dir=Path(args.output_dir),
        ),
    )

    logger.info(f"Configuration:")
    logger.info(f"  Source: {config.data.source_path}")
    logger.info(f"  Runs: {config.experiment.n_runs}")
    logger.info(f"  Output: {config.output.output_dir}")

    data_loader = AnalysisDataLoader(
        config.data.source_path,
        config.data.csv_filename,
        logger=logger,
    )
    data_loader.extract_and_validate()
    data_loader.load_full_dataset()

    data_profile = None
    if not args.skip_eda:
        data_profile = run_eda(config, data_loader, logger)

    seed_manager = SeedManager(config.experiment.master_seed, logger)
    seed_manager.generate_sequence(config.experiment.n_runs)

    if args.skip_training and args.results_dir:
        logger.info("Loading existing results...")
        aggregator = ResultsAggregator(logger)
        comparative_result = aggregator.load_results(Path(args.results_dir) / "comparative_result.json")
    else:
        comparative_result = run_experiments(config, seed_manager, logger)

    statistical_result = run_statistical_analysis(comparative_result, config, logger)

    figures_dir = run_visualization(comparative_result, statistical_result, config, logger)

    run_reporting(
        comparative_result,
        statistical_result,
        data_profile,
        figures_dir,
        config,
        logger,
    )

    logger.info("=" * 60)
    logger.info("Analysis Complete!")
    logger.info(f"Results saved to: {config.output.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
