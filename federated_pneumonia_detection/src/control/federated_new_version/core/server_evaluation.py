"""
Server-side centralized evaluation for federated learning.

This module provides a centralized evaluation function that evaluates the
aggregated global model on a server-side test dataset after each round.
"""

import torch
from flwr.app import ArrayRecord, MetricRecord
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import (
    XRayDataModule,
)
import pandas as pd
from logging import getLogger

logger = getLogger(__name__)


def create_central_evaluate_fn(
    config_manager: ConfigManager, csv_path: str, image_dir: str
):
    """
    Factory function to create a centralized evaluation function.

    Args:
        config_manager: Configuration manager for model initialization
        csv_path: Path to CSV file with test data
        image_dir: Directory containing test images

    Returns:
        Callable evaluation function for server-side evaluation
    """

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """
        Evaluate the aggregated global model on server-side test dataset.

        This function is called by the strategy after each round of federated learning.
        It performs centralized evaluation on a held-out test set.

        Args:
            server_round: Current round number of federated learning
            arrays: Aggregated model parameters from the current round

        Returns:
            MetricRecord containing evaluation metrics (loss, accuracy, etc.)
        """
        logger.info(
            f"[Server Evaluation] Starting centralized evaluation for round {server_round}"
        )

        # Load model and initialize with aggregated weights
        model = LitResNet(config=config_manager)
        model.load_state_dict(arrays.to_torch_state_dict())

        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Load test dataset
        df = pd.read_csv(csv_path)

        # Split data - use a portion for server-side evaluation
        # Use the last 20% as server test set (different from client partitions)
        test_size = int(len(df) * 0.2)
        test_df = df.iloc[-test_size:].reset_index(drop=True)

        # Add filename column if missing (required by XRayDataModule)
        if "filename" not in test_df.columns:
            if "patientId" in test_df.columns:
                test_df["filename"] = test_df["patientId"].astype(str) + ".png"
            else:
                raise ValueError(
                    "DataFrame must contain either 'filename' or 'patientId' column"
                )

        logger.info(f"[Server Evaluation] Using {len(test_df)} samples for evaluation")

        # Create data module for server evaluation
        data_module = XRayDataModule(
            train_df=test_df,  # Dummy, won't be used
            val_df=test_df,  # Use for evaluation
            config=config_manager,
            image_dir=image_dir,
        )
        # Setup for validation stage (not test) to access val_dataloader
        data_module.setup(stage="validate")
        test_loader = data_module.val_dataloader()

        # Evaluate model
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

                # Forward pass
                logits = model(images)
                loss = model._calculate_loss(logits, labels)

                # Get predictions
                preds = model._get_predictions(logits)
                targets = model._prepare_targets_for_metrics(labels)

                # Accumulate metrics
                total_loss += loss.item() * len(labels)
                total_correct += (preds == targets).sum().item()
                total_samples += len(labels)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Calculate additional metrics using torchmetrics
        from torchmetrics import Precision, Recall, F1Score, AUROC

        preds_tensor = torch.tensor(all_preds)
        targets_tensor = torch.tensor(all_targets)

        precision_metric = Precision(task="binary")
        recall_metric = Recall(task="binary")
        f1_metric = F1Score(task="binary")
        auroc_metric = AUROC(task="binary")

        precision = precision_metric(preds_tensor, targets_tensor).item()
        recall = recall_metric(preds_tensor, targets_tensor).item()
        f1 = f1_metric(preds_tensor, targets_tensor).item()
        auroc = auroc_metric(preds_tensor, targets_tensor).item()

        logger.info(
            f"[Server Evaluation] Round {server_round} - "
            f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
            f"Prec: {precision:.4f}, Rec: {recall:.4f}, "
            f"F1: {f1:.4f}, AUROC: {auroc:.4f}"
        )

        # Return metrics as MetricRecord
        return MetricRecord(
            {
                "server_loss": avg_loss,
                "server_accuracy": accuracy,
                "server_precision": precision,
                "server_recall": recall,
                "server_f1": f1,
                "server_auroc": auroc,
            }
        )

    return central_evaluate
