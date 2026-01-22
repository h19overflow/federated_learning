"""
Server-side centralized evaluation for federated learning.

This module provides a centralized evaluation function that evaluates the
aggregated global model on a server-side test dataset after each round.
"""

from logging import getLogger

import pandas as pd
import torch
from flwr.app import ArrayRecord, MetricRecord

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.xray_data_module import (
    XRayDataModule,
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
            f"[Server Evaluation] Starting centralized evaluation for round {server_round}",
        )

        model = LitResNet(config=config_manager)
        model.load_state_dict(arrays.to_torch_state_dict())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

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

        data_module = XRayDataModule(
            train_df=test_df,
            val_df=test_df,
            config=config_manager,
            image_dir=image_dir,
        )
        data_module.setup(stage="validate")
        test_loader = data_module.val_dataloader()

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

        from torchmetrics import AUROC, ConfusionMatrix, F1Score, Precision, Recall

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

        logger.info(
            f"[Server Evaluation] Round {server_round} - "
            f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
            f"Prec: {precision:.4f}, Rec: {recall:.4f}, "
            f"F1: {f1:.4f}, AUROC: {auroc:.4f}, "
            f"CM(TP/TN/FP/FN): {tp}/{tn}/{fp}/{fn}",
        )

        return MetricRecord(
            {
                "server_loss": avg_loss,
                "server_accuracy": accuracy,
                "server_precision": precision,
                "server_recall": recall,
                "server_f1": f1,
                "server_auroc": auroc,
                "server_cm_tp": float(tp),
                "server_cm_tn": float(tn),
                "server_cm_fp": float(fp),
                "server_cm_fn": float(fn),
            },
        )

    return central_evaluate
