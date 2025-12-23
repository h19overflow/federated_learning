"""
Baseline training script to establish current model performance.
This script runs the centralized trainer with default configuration.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer
from federated_pneumonia_detection.config.config_manager import ConfigManager


def run_baseline_training():
    """Run baseline training to establish current metrics."""

    # Configuration
    source_path = r"C:\Users\User\Projects\FYP2\Training_Sample_5pct.zip"
    output_dir = project_root / "model_enhancement_results" / "baseline"

    # Create output directories
    checkpoint_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BASELINE TRAINING - Establishing Current Metrics")
    print("=" * 60)

    # Initialize trainer
    trainer = CentralizedTrainer(
        checkpoint_dir=str(checkpoint_dir),
        logs_dir=str(logs_dir),
    )

    # Log configuration
    config = trainer.config
    print(f"\nConfiguration:")
    print(f"  - Epochs: {config.get('experiment.epochs')}")
    print(f"  - Batch Size: {config.get('experiment.batch_size')}")
    print(f"  - Learning Rate: {config.get('experiment.learning_rate')}")
    print(f"  - Dropout Rate: {config.get('experiment.dropout_rate')}")
    print(f"  - Fine-tune Layers: {config.get('experiment.fine_tune_layers_count')}")
    print(f"  - Custom Preprocessing: {config.get('experiment.use_custom_preprocessing')}")

    # Run training
    print("\nStarting training...")
    results = trainer.train(
        source_path=source_path,
        experiment_name="baseline_run",
        run_id=1,
    )

    # Save results
    results_path = output_dir / "baseline_results.json"

    # Make results JSON serializable
    serializable_results = {
        "best_model_path": results.get("best_model_path"),
        "best_model_score": results.get("best_model_score"),
        "current_epoch": results.get("current_epoch"),
        "global_step": results.get("global_step"),
        "state": results.get("state"),
        "total_epochs_trained": results.get("total_epochs_trained"),
        "metrics_history": results.get("metrics_history", []),
    }

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)

    metrics_history = results.get("metrics_history", [])
    if metrics_history:
        last_epoch = metrics_history[-1]
        print(f"\nFinal Epoch Metrics:")
        print(f"  - Validation Accuracy: {last_epoch.get('val_acc', 'N/A'):.4f}")
        print(f"  - Validation Precision: {last_epoch.get('val_precision', 'N/A'):.4f}")
        print(f"  - Validation Recall: {last_epoch.get('val_recall', 'N/A'):.4f}")
        print(f"  - Validation F1: {last_epoch.get('val_f1', 'N/A'):.4f}")
        print(f"  - Validation AUROC: {last_epoch.get('val_auroc', 'N/A'):.4f}")
        print(f"  - Validation Loss: {last_epoch.get('val_loss', 'N/A'):.4f}")

    print(f"\nBest Model Score: {results.get('best_model_score', 'N/A')}")
    print(f"Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_baseline_training()
