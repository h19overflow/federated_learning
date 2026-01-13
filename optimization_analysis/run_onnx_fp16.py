"""
ONNX FP16 benchmark runner.
Runs ONNX FP16 inference on dataset and generates comprehensive results.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_analysis/data/onnx_fp16/benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization_analysis import BenchmarkSuite
from optimization_analysis.inference_wrappers.onnx_inference import ONNXFP16InferenceWrapper
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_labels_from_directory(image_dir: str) -> np.ndarray:
    """Create true labels based on directory structure."""
    data_path = Path(image_dir)
    labels = []

    normal_dir = data_path / "normal"
    pneumonia_dir = data_path / "pneumonia"

    normal_count = len(list(normal_dir.glob("*.png")))
    pneumonia_count = len(list(pneumonia_dir.glob("*.png")))

    logger.info(f"NORMAL images: {normal_count}")
    logger.info(f"PNEUMONIA images: {pneumonia_count}")

    labels.extend([0] * normal_count)
    labels.extend([1] * pneumonia_count)

    return np.array(labels)


def plot_timing_distribution(stage_stats: dict, output_dir: Path):
    """Plot timing distribution for each stage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ONNX FP16 Inference Timing Distribution', fontsize=16, fontweight='bold')

    stages = ['preprocessing', 'feature_extraction', 'classification']
    stage_names = {
        'preprocessing': 'Preprocessing',
        'feature_extraction': 'Feature Extraction',
        'classification': 'Classification'
    }

    for idx, stage in enumerate(stages):
        if stage not in stage_stats:
            continue

        stats = stage_stats[stage]
        times = [stats.get('mean', 0)] * stats.get('count', 1)

        ax = axes[idx // 2, idx % 2]
        ax.barh([0], [stats['mean']], color='steelblue', alpha=0.7, label='Mean')
        ax.barh([1], [stats['median']], color='forestgreen', alpha=0.7, label='Median')
        ax.barh([2], [stats['p95']], color='darkorange', alpha=0.7, label='p95')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Mean', 'Median', 'p95'])
        ax.set_xlabel('Time (ms)')
        ax.set_title(f'{stage_names[stage]}', fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

    if 'total' in stage_stats:
        total_stats = stage_stats['total']
        ax = axes[1, 0]
        metrics = ['Mean', 'Median', 'p95', 'p99']
        values = [total_stats.get('mean', 0),
                 total_stats.get('median', 0),
                 total_stats.get('p95', 0),
                 total_stats.get('p99', 0)]
        colors = ['steelblue', 'forestgreen', 'darkorange', 'crimson']

        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_ylabel('Time (ms)')
        ax.set_title('Total Inference Time', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=10)

    ax = axes[1, 1]
    stages_data = [stage_stats.get(s, {}).get('mean', 0) for s in stages]
    stage_labels = [stage_names[s] for s in stages]

    bars = ax.barh(stage_labels, stages_data, color=['steelblue', 'forestgreen', 'darkorange'])
    ax.set_xlabel('Time (ms)')
    ax.set_title('Stage-wise Timing Comparison', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, value in zip(bars, stages_data):
        ax.text(value + 0.5, bar.get_y() + bar.get_height()/2.,
               f'{value:.2f}ms', va='center', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'timing_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved timing distribution plot: {output_path}")


def plot_confusion_matrix(metrics: dict, output_dir: Path):
    """Plot confusion matrix."""
    if not metrics:
        return

    cm = np.array([
        [metrics.get('true_negatives', 0), metrics.get('false_positives', 0)],
        [metrics.get('false_negatives', 0), metrics.get('true_positives', 0)]
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                ax=ax, square=True, vmin=0, vmax=cm.max())

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - ONNX FP16', fontsize=14, fontweight='bold')

    ax.set_xticklabels(['NORMAL', 'PNEUMONIA'], rotation=0, ha='center')
    ax.set_yticklabels(['NORMAL', 'PNEUMONIA'], rotation=0, va='center')

    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix: {output_path}")


def plot_accuracy_metrics(metrics: dict, output_dir: Path):
    """Plot accuracy metrics comparison."""
    if not metrics:
        return

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1', 0),
        metrics.get('auroc', 0)
    ]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics - ONNX FP16', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')
    ax.legend()

    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    output_path = output_dir / 'accuracy_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved accuracy metrics plot: {output_path}")


def plot_stage_percentiles(stage_stats: dict, output_dir: Path):
    """Plot percentiles for each stage."""
    stages = ['preprocessing', 'feature_extraction', 'classification', 'total']
    stage_names = {
        'preprocessing': 'Preprocessing',
        'feature_extraction': 'Feature Extraction',
        'classification': 'Classification',
        'total': 'Total'
    }

    percentiles = ['mean', 'median', 'p50', 'p95', 'p99']
    percentile_labels = ['Mean', 'Median', 'p50', 'p95', 'p99']

    data = []
    labels = []

    for stage in stages:
        if stage in stage_stats:
            stats = stage_stats[stage]
            stage_data = [stats.get(p, 0) for p in percentiles]
            data.append(stage_data)
            labels.append(stage_names[stage])

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(percentiles))
    width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(len(stages)):
        ax.bar(x + i * width, data[i], width, label=labels[i], color=colors[i], alpha=0.8)

    ax.set_xlabel('Percentile', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Stage-wise Percentile Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(percentile_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'stage_percentiles.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved stage percentiles plot: {output_path}")


def save_summary_report(result, output_dir: Path, execution_provider: str = None, precision_mode: str = None):
    """Save comprehensive summary report."""
    report = {
        "benchmark_summary": {
            "approach": result.approach_name,
            "timestamp": result.timestamp,
            "num_samples": result.num_samples,
            "execution_provider": execution_provider,
            "precision_mode": precision_mode
        },
        "timing_results": {
            "total_avg_ms": result.total_time_avg,
            "total_p50_ms": result.stage_stats.get('total', {}).get('p50', 0),
            "total_p95_ms": result.stage_stats.get('total', {}).get('p95', 0),
            "total_p99_ms": result.stage_stats.get('total', {}).get('p99', 0),
            "stages": {
                "preprocessing": {
                    "avg_ms": result.stage_stats.get('preprocessing', {}).get('mean', 0),
                    "p50_ms": result.stage_stats.get('preprocessing', {}).get('p50', 0),
                    "p95_ms": result.stage_stats.get('preprocessing', {}).get('p95', 0),
                    "p99_ms": result.stage_stats.get('preprocessing', {}).get('p99', 0),
                    "percentage": (result.stage_stats.get('preprocessing', {}).get('mean', 0) / result.total_time_avg * 100)
                },
                "feature_extraction": {
                    "avg_ms": result.stage_stats.get('feature_extraction', {}).get('mean', 0),
                    "p50_ms": result.stage_stats.get('feature_extraction', {}).get('p50', 0),
                    "p95_ms": result.stage_stats.get('feature_extraction', {}).get('p95', 0),
                    "p99_ms": result.stage_stats.get('feature_extraction', {}).get('p99', 0),
                    "percentage": (result.stage_stats.get('feature_extraction', {}).get('mean', 0) / result.total_time_avg * 100)
                },
                "classification": {
                    "avg_ms": result.stage_stats.get('classification', {}).get('mean', 0),
                    "p50_ms": result.stage_stats.get('classification', {}).get('p50', 0),
                    "p95_ms": result.stage_stats.get('classification', {}).get('p95', 0),
                    "p99_ms": result.stage_stats.get('classification', {}).get('p99', 0),
                    "percentage": (result.stage_stats.get('classification', {}).get('mean', 0) / result.total_time_avg * 100)
                }
            }
        },
        "accuracy_metrics": result.accuracy_metrics
    }

    output_path = output_dir / 'onnx_fp16_report.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved JSON report: {output_path}")

    txt_output_path = output_dir / 'onnx_fp16_report.txt'
    with open(txt_output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ONNX FP16 BENCHMARK REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Approach: {result.approach_name}\n")
        f.write(f"Timestamp: {result.timestamp}\n")
        f.write(f"Samples: {result.num_samples}\n")

        if execution_provider:
            f.write(f"Execution Provider: {execution_provider}\n")
        if precision_mode:
            f.write(f"Precision Mode: {precision_mode}\n")
        f.write("\n")

        f.write("-"*80 + "\n")
        f.write("TIMING RESULTS\n")
        f.write("-"*80 + "\n\n")

        f.write(f"Total Average Time: {result.total_time_avg:.2f} ms\n")
        f.write(f"Total Median Time: {result.stage_stats.get('total', {}).get('median', 0):.2f} ms\n")
        f.write(f"Total p95 Time: {result.stage_stats.get('total', {}).get('p95', 0):.2f} ms\n")
        f.write(f"Total p99 Time: {result.stage_stats.get('total', {}).get('p99', 0):.2f} ms\n\n")

        f.write("Stage Breakdown:\n")
        for stage in ['preprocessing', 'feature_extraction', 'classification']:
            stats = result.stage_stats.get(stage, {})
            if stats:
                f.write(f"\n{stage.upper()}:\n")
                f.write(f"  Average: {stats.get('mean', 0):.2f} ms\n")
                f.write(f"  Median:  {stats.get('median', 0):.2f} ms\n")
                f.write(f"  p95:      {stats.get('p95', 0):.2f} ms\n")
                f.write(f"  p99:      {stats.get('p99', 0):.2f} ms\n")
                f.write(f"  StdDev:   {stats.get('stddev', 0):.2f} ms\n")
                f.write(f"  Percentage: {(stats.get('mean', 0) / result.total_time_avg * 100):.1f}%\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("ACCURACY METRICS\n")
        f.write("-"*80 + "\n\n")

        if result.accuracy_metrics:
            f.write(f"Accuracy:    {result.accuracy_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision:   {result.accuracy_metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:      {result.accuracy_metrics.get('recall', 0):.4f}\n")
            f.write(f"F1 Score:    {result.accuracy_metrics.get('f1', 0):.4f}\n")
            f.write(f"AUROC:       {result.accuracy_metrics.get('auroc', 'N/A')}\n\n")

            f.write("Confusion Matrix:\n")
            f.write(f"  True Negatives:  {result.accuracy_metrics.get('true_negatives', 0)}\n")
            f.write(f"  False Positives: {result.accuracy_metrics.get('false_positives', 0)}\n")
            f.write(f"  False Negatives: {result.accuracy_metrics.get('false_negatives', 0)}\n")
            f.write(f"  True Positives:  {result.accuracy_metrics.get('true_positives', 0)}\n\n")

            f.write(f"Specificity: {result.accuracy_metrics.get('specificity', 0):.4f}\n")
            f.write(f"Sensitivity: {result.accuracy_metrics.get('sensitivity', 0):.4f}\n")

    logger.info(f"Saved text report: {txt_output_path}")


def main():
    """Run ONNX FP16 benchmark and generate comprehensive results."""
    logger.info("="*80)
    logger.info("STARTING ONNX FP16 BENCHMARK")
    logger.info("="*80)

    image_dir = r"C:\Users\User\Projects\FYP2\data\Training_Sample_5pct\organized_images"
    checkpoint_path = r"federated_pneumonia_detection\src\control\model_inferance\pneumonia_model_07_0.928.ckpt"
    output_dir = Path("optimization_analysis/data/onnx_fp16")

    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    logger.info("\nCreating labels from directory structure...")
    labels = create_labels_from_directory(image_dir)

    # Initialize benchmark suite
    logger.info("\nInitializing benchmark suite...")
    suite = BenchmarkSuite(
        image_dir=image_dir,
        checkpoint_path=checkpoint_path,
        num_warmup=10,
        num_samples=1334
    )

    logger.info("\nLoading images...")
    images = suite.load_images()
    logger.info(f"Loaded {len(images)} images")

    if len(images) == 0:
        logger.error("No images loaded! Exiting.")
        return

    logger.info("\nCreating ONNX FP16 inference wrapper...")

    try:
        from optimization_analysis.inference_wrappers.onnx_inference import ONNXFP16InferenceWrapper
        logger.info("Using ONNXFP16InferenceWrapper with FP16 optimization")

        wrapper = ONNXFP16InferenceWrapper(checkpoint_path=checkpoint_path)

        execution_provider = None
        precision_mode = None

        if hasattr(wrapper, 'session'):
            providers = wrapper.session.get_providers()
            logger.info(f"Execution Providers: {providers}")

            if 'CUDAExecutionProvider' in providers:
                execution_provider = "CUDAExecutionProvider"
            elif 'CPUExecutionProvider' in providers:
                execution_provider = "CPUExecutionProvider"
            else:
                execution_provider = providers[0] if providers else "Unknown"

            logger.info(f"Execution Provider: {execution_provider}")

        if hasattr(wrapper, 'get_precision_mode'):
            precision_mode = wrapper.get_precision_mode()
            logger.info(f"Precision Mode: {precision_mode}")
        else:
            precision_mode = "Unknown"
            logger.warning("Could not determine precision mode")

        logger.info("="*80)
        logger.info("FP16 CONFIGURATION:")
        logger.info(f"FP16 Enabled: {getattr(wrapper, 'fp16_enabled', 'Unknown')}")
        logger.info(f"Execution Provider: {getattr(wrapper, 'execution_provider_used', 'Unknown')}")
        logger.info("="*80)
        logger.warning("FP16 PRECISION NOTE:")
        logger.warning("The ONNX FP16 wrapper is currently being developed.")
        logger.warning("This script uses ONNXInferenceWrapper (FP32) as a placeholder.")
        logger.warning("Once ONNXFP16InferenceWrapper is available, this script will use it.")
        logger.warning("="*80)

    except ImportError as e:
        logger.error(f"Failed to import ONNX inference wrapper: {e}")
        logger.error("Please ensure ONNX inference wrapper is available.")
        return

    logger.info("\n" + "="*80)
    logger.info("RUNNING BENCHMARK")
    logger.info("="*80)

    result = suite.run_benchmark(wrapper, images, labels)

    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)

    logger.info(f"\nApproach: {result.approach_name}")
    logger.info(f"Samples: {result.num_samples}")
    logger.info(f"Total Time: {result.total_time_avg:.2f} ms")
    logger.info(f"Total p50: {result.stage_stats.get('total', {}).get('p50', 0):.2f} ms")
    logger.info(f"Total p95: {result.stage_stats.get('total', {}).get('p95', 0):.2f} ms")
    logger.info(f"Total p99: {result.stage_stats.get('total', {}).get('p99', 0):.2f} ms")

    if execution_provider:
        logger.info(f"Execution Provider: {execution_provider}")
    if precision_mode:
        logger.info(f"Precision Mode: {precision_mode}")

    logger.info("\nStage Breakdown:")
    for stage in ['preprocessing', 'feature_extraction', 'classification']:
        stats = result.stage_stats.get(stage, {})
        if stats:
            logger.info(f"  {stage:15s}: {stats.get('mean', 0):.2f} ms "
                       f"(p95: {stats.get('p95', 0):.2f} ms, "
                       f"{(stats.get('mean', 0) / result.total_time_avg * 100):.1f}%)")

    if result.accuracy_metrics:
        logger.info("\nAccuracy Metrics:")
        logger.info(f"  Accuracy:    {result.accuracy_metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Precision:   {result.accuracy_metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall:      {result.accuracy_metrics.get('recall', 0):.4f}")
        logger.info(f"  F1 Score:    {result.accuracy_metrics.get('f1', 0):.4f}")
        logger.info(f"  AUROC:       {result.accuracy_metrics.get('auroc', 'N/A')}")

        logger.info("\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {result.accuracy_metrics.get('true_negatives', 0)}")
        logger.info(f"  False Positives: {result.accuracy_metrics.get('false_positives', 0)}")
        logger.info(f"  False Negatives: {result.accuracy_metrics.get('false_negatives', 0)}")
        logger.info(f"  True Positives:  {result.accuracy_metrics.get('true_positives', 0)}")

    logger.info("\n" + "="*80)
    logger.info("GENERATING PLOTS")
    logger.info("="*80)

    logger.info("\nGenerating timing distribution plot...")
    plot_timing_distribution(result.stage_stats, output_dir)

    logger.info("Generating confusion matrix plot...")
    plot_confusion_matrix(result.accuracy_metrics, output_dir)

    logger.info("Generating accuracy metrics plot...")
    plot_accuracy_metrics(result.accuracy_metrics, output_dir)

    logger.info("Generating stage percentiles plot...")
    plot_stage_percentiles(result.stage_stats, output_dir)

    logger.info("\n" + "="*80)
    logger.info("SAVING REPORTS")
    logger.info("="*80)

    save_summary_report(result, output_dir, execution_provider, precision_mode)

    logger.info("\n" + "="*80)
    logger.info("ONNX FP16 BENCHMARK COMPLETE")
    logger.info("="*80)

    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info(f"  - {output_dir / 'onnx_fp16_report.json'}")
    logger.info(f"  - {output_dir / 'onnx_fp16_report.txt'}")
    logger.info(f"  - {output_dir / 'timing_distribution.png'}")
    logger.info(f"  - {output_dir / 'confusion_matrix.png'}")
    logger.info(f"  - {output_dir / 'accuracy_metrics.png'}")
    logger.info(f"  - {output_dir / 'stage_percentiles.png'}")
    logger.info(f"  - {output_dir / 'benchmark.log'}")

    logger.info("\n" + "="*80)
    logger.info("FP16 OPTIMIZATION SUMMARY")
    logger.info("="*80)
    if execution_provider == "CUDAExecutionProvider":
        logger.info("✓ FP16 optimization enabled on CUDA GPU")
        logger.info("  - Execution Provider: CUDAExecutionProvider")
        logger.info("  - Precision Mode: FP16")
        logger.info("  - Model is using half-precision for improved performance")
    else:
        logger.info("⚠ FP16 running on CPU (no speedup expected)")
        logger.info("  - Execution Provider: CPUExecutionProvider")
        logger.info("  - Note: FP16 provides speedup only on CUDA GPUs")
        logger.info("  - For GPU acceleration, install onnxruntime-gpu")
    logger.info("="*80)



if __name__ == "__main__":
    main()
