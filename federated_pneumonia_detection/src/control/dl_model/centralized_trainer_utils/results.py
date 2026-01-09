"""Training results collection and persistence."""

import os
import json
import logging
from typing import Optional, Dict, Any

import pytorch_lightning as pl

from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import (
    LitResNet,
)


def collect_training_results(
    trainer: pl.Trainer,
    model: LitResNet,
    metrics_collector: Any,
    logs_dir: str,
    checkpoint_dir: str,
    logger: logging.Logger,
    run_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Collect and organize training results.

    Args:
        trainer: Trained PyTorch Lightning trainer
        model: Trained model instance
        metrics_collector: MetricsCollectorCallback instance
        logs_dir: Directory for training logs
        checkpoint_dir: Directory for model checkpoints
        logger: Logger instance
        run_id: Optional database run ID

    Returns:
        Dictionary with training results
    """
    checkpoint_callback = None
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            checkpoint_callback = callback
            break

    best_model_path = None
    best_model_score = None

    if checkpoint_callback:
        best_model_path = (
            checkpoint_callback.best_model_path
            if checkpoint_callback.best_model_path
            else None
        )
        if checkpoint_callback.best_model_score is not None:
            best_model_score = (
                checkpoint_callback.best_model_score.item()
                if hasattr(checkpoint_callback.best_model_score, "item")
                else float(checkpoint_callback.best_model_score)
            )

    state_value = None
    if trainer.state and hasattr(trainer.state, "stage") and trainer.state.stage:
        state_value = (
            trainer.state.stage.value
            if hasattr(trainer.state.stage, "value")
            else str(trainer.state.stage)
        )

    metrics_history = (
        metrics_collector.get_metrics_history() if metrics_collector else []
    )
    metrics_metadata = metrics_collector.get_metadata() if metrics_collector else {}

    results = {
        "run_id": run_id,
        "best_model_path": best_model_path,
        "best_model_score": best_model_score,
        "current_epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "state": state_value,
        "model_summary": model.get_model_summary(),
        "checkpoint_dir": checkpoint_dir,
        "logs_dir": logs_dir,
        "metrics_history": metrics_history,
        "metrics_metadata": metrics_metadata,
        "total_epochs_trained": len(metrics_history),
    }

    logger.info(
        f"Training results collected: {len(metrics_history)} epochs tracked"
    )

    logger.info(
        f"Saving metrics to: results/centralized/metrics_output/metrics.json"
    )

    output_path = os.path.join(logs_dir, "metrics_output", "metrics.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)

    return results
