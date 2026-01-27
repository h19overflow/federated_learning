"""
Server-side centralized evaluation for federated learning.

This module provides a centralized evaluation function that evaluates the
aggregated global model on a server-side test dataset after each round.
"""

from logging import getLogger
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from flwr.app import ArrayRecord, MetricRecord
from torchmetrics import AUROC, ConfusionMatrix, F1Score, Precision, Recall

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.internals.data.xray_data_module import (  # noqa: E501
    XRayDataModule,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced import (  # noqa: E501
    LitResNetEnhanced,
)

logger = getLogger(__name__)


def create_central_evaluate_fn(
    config_manager: ConfigManager,
    csv_path: str,
    image_dir: str,
):
    """Factory function to create a centralized evaluation function.

    Args:
        config_manager: Configuration manager for model initialization
        csv_path: Path to CSV file with test data
        image_dir: Directory containing test images

    Returns:
        Callable evaluation function for server-side evaluation
    """

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate the aggregated global model on server-side test dataset."""
        logger.info(
            f"[Server Evaluation] Starting centralized evaluation for round {server_round}",  # noqa: E501
        )

        # 1. Load and prepare model
        model, device = _load_and_prepare_model(config_manager, arrays)

        # 2. Load and prepare test data
        test_df = _load_and_prepare_test_data(csv_path)

        # 3. Create dataloader
        test_loader = _create_dataloader(test_df, config_manager, image_dir)

        # 4. Run inference loop
        avg_loss, accuracy, all_preds, all_targets = _run_inference_loop(
            model, test_loader, device
        )

        # 5. Calculate advanced metrics
        metrics = _calculate_advanced_metrics(all_preds, all_targets)

        # 6. Log results
        logger.info(
            f"[Server Evaluation] Round {server_round} - "
            f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
            f"Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}, AUROC: {metrics['auroc']:.4f}, "
            f"CM(TP/TN/FP/FN): {metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}",  # noqa: E501
        )

        # 7. Return metrics
        return MetricRecord(
            {
                "server_loss": avg_loss,
                "server_accuracy": accuracy,
                "server_precision": metrics["precision"],
                "server_recall": metrics["recall"],
                "server_f1": metrics["f1"],
                "server_auroc": metrics["auroc"],
                "server_cm_tp": float(metrics["tp"]),
                "server_cm_tn": float(metrics["tn"]),
                "server_cm_fp": float(metrics["fp"]),
                "server_cm_fn": float(metrics["fn"]),
            },
        )

    return central_evaluate


def _load_and_prepare_model(
    config_manager: ConfigManager,
    arrays: ArrayRecord,
) -> Tuple[LitResNetEnhanced, torch.device]:
    """Load model from aggregated weights and prepare for evaluation.

    Args:
        config_manager: Configuration manager for model initialization
        arrays: Aggregated model weights from server

    Returns:
        Tuple of (model, device)
    """
    model = LitResNetEnhanced(
        config=config_manager, use_focal_loss=False, use_cosine_scheduler=False
    )
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, device


def _load_and_prepare_test_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and prepare test dataset (last 20% of data).

    Args:
        csv_path: Path to CSV file with data

    Returns:
        DataFrame with test data and filename column
    """
    df = pd.read_csv(csv_path)

    test_size = int(len(df) * 0.2)
    test_df = df.iloc[-test_size:].reset_index(drop=True)

    if "filename" not in test_df.columns:
        if "patientId" in test_df.columns:
            test_df["filename"] = test_df["patientId"].astype(str) + ".png"
        else:
            raise ValueError(
                "DataFrame must contain either 'filename' or 'patientId' column",
            )

    logger.info(f"[Server Evaluation] Using {len(test_df)} samples for evaluation")

    return test_df


def _create_dataloader(
    test_df: pd.DataFrame,
    config_manager: ConfigManager,
    image_dir: str,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for test dataset.

    Args:
        test_df: Test DataFrame
        config_manager: Configuration manager
        image_dir: Directory containing images

    Returns:
        DataLoader for test data
    """
    data_module = XRayDataModule(
        train_df=test_df,
        val_df=test_df,
        config=config_manager,
        image_dir=image_dir,
    )
    data_module.setup(stage="validate")
    return data_module.val_dataloader()


def _run_inference_loop(
    model: LitResNetEnhanced,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Run inference on test data and accumulate predictions.

    Args:
        model: Model in eval mode
        test_loader: DataLoader with test batches
        device: Device to run inference on

    Returns:
        Tuple of (avg_loss, accuracy, all_preds, all_targets)
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = model._calculate_loss(logits, labels)

            preds = model._get_predictions(logits)
            targets = model._prepare_targets_for_metrics(labels)

            if model.num_classes == 1:
                preds_binary = (preds > 0.5).int().squeeze()
                targets_flat = targets.squeeze()
            else:
                preds_binary = preds.argmax(dim=1)
                targets_flat = targets

            total_loss += loss.item() * len(labels)
            total_correct += (preds_binary == targets_flat).sum().item()
            total_samples += len(labels)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)


def _calculate_advanced_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
) -> Dict[str, Any]:
    """Calculate advanced metrics (Precision, Recall, F1, AUROC, CM).

    Args:
        all_preds: Array of predictions
        all_targets: Array of targets

    Returns:
        Dictionary with metric names and values
    """
    preds_tensor = torch.tensor(all_preds)
    targets_tensor = torch.tensor(all_targets)

    precision_metric = Precision(task="binary")
    recall_metric = Recall(task="binary")
    f1_metric = F1Score(task="binary")
    auroc_metric = AUROC(task="binary")
    cm_metric = ConfusionMatrix(task="binary")

    precision = precision_metric(preds_tensor, targets_tensor).item()
    recall = recall_metric(preds_tensor, targets_tensor).item()
    f1 = f1_metric(preds_tensor, targets_tensor).item()
    auroc = auroc_metric(preds_tensor, targets_tensor).item()

    cm = cm_metric(preds_tensor, targets_tensor)
    tn = int(cm[0, 0].item())
    fp = int(cm[0, 1].item())
    fn = int(cm[1, 0].item())
    tp = int(cm[1, 1].item())

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
