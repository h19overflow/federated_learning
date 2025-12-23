"""
Test the best model on the full dataset.
Evaluates recall, precision, F1, accuracy, and AUROC.
"""

import sys
import json
from pathlib import Path
import logging

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet_enhanced import LitResNetEnhanced
from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import XRayDataModule
from federated_pneumonia_detection.src.control.dl_model.utils import DataSourceExtractor
from federated_pneumonia_detection.src.utils.data_processing import load_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    checkpoint_path: str,
    source_path: str,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate a trained model on the full dataset.

    Args:
        checkpoint_path: Path to the model checkpoint
        source_path: Path to the dataset (zip or directory)
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with all metrics
    """
    logger.info("=" * 70)
    logger.info("MODEL EVALUATION ON FULL DATASET")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Setup
    config = ConfigManager()
    config.set("system.sample_fraction", 1.0)  # Use FULL dataset
    config.set("experiment.batch_size", batch_size)

    # Extract data
    logger.info("\n[1/4] Loading full dataset...")
    data_extractor = DataSourceExtractor(logger)
    image_dir, csv_path = data_extractor.extract_and_validate(source_path)

    # Load ALL data (no split - evaluate on everything)
    df = load_metadata(csv_path, config, logger)
    logger.info(f"Total samples in dataset: {len(df)}")

    # Class distribution
    target_col = config.get("columns.target", "Target")
    pos_count = (df[target_col] == 1).sum()
    neg_count = (df[target_col] == 0).sum()
    logger.info(f"Class distribution: {neg_count} negative, {pos_count} positive ({pos_count/len(df):.1%} positive)")

    # Create data module with all data as "validation"
    logger.info("\n[2/4] Creating data loader...")
    data_module = XRayDataModule(
        train_df=df.head(1),  # Dummy train (not used)
        val_df=df,  # Full dataset for evaluation
        config=config,
        image_dir=image_dir,
        validate_images_on_init=False,
    )
    data_module.setup("validate")
    val_loader = data_module.val_dataloader()

    # Load model
    logger.info("\n[3/4] Loading model from checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = LitResNetEnhanced.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=device,
    )
    model.to(device)
    model.eval()

    # Evaluate
    logger.info("\n[4/4] Running evaluation...")
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images, targets = batch
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.numpy().flatten())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auroc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)

    # Results
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auroc": float(auroc),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "total_samples": len(all_targets),
        "positive_samples": int(all_targets.sum()),
        "negative_samples": int(len(all_targets) - all_targets.sum()),
    }

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"\nDataset: {results['total_samples']} samples")
    logger.info(f"  - Negative (healthy): {results['negative_samples']}")
    logger.info(f"  - Positive (pneumonia): {results['positive_samples']}")

    logger.info(f"\n{'='*40}")
    logger.info(f"  ACCURACY:   {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  PRECISION:  {precision:.4f} ({precision*100:.2f}%)")
    logger.info(f"  RECALL:     {recall:.4f} ({recall*100:.2f}%)")
    logger.info(f"  F1 SCORE:   {f1:.4f} ({f1*100:.2f}%)")
    logger.info(f"  AUROC:      {auroc:.4f} ({auroc*100:.2f}%)")
    logger.info(f"{'='*40}")

    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                 Predicted")
    logger.info(f"              Neg      Pos")
    logger.info(f"  Actual Neg  {cm[0,0]:5d}    {cm[0,1]:5d}  (FP rate: {cm[0,1]/(cm[0,0]+cm[0,1])*100:.1f}%)")
    logger.info(f"  Actual Pos  {cm[1,0]:5d}    {cm[1,1]:5d}  (FN rate: {cm[1,0]/(cm[1,0]+cm[1,1])*100:.1f}%)")

    logger.info(f"\nMedical Interpretation:")
    logger.info(f"  - Sensitivity (Recall): {recall*100:.1f}% of pneumonia cases correctly identified")
    logger.info(f"  - Specificity: {cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}% of healthy cases correctly identified")
    logger.info(f"  - False Negatives (missed pneumonia): {cm[1,0]} cases")
    logger.info(f"  - False Positives (false alarms): {cm[0,1]} cases")

    return results


def main():
    # Best v2 model with highest recall
    checkpoint_path = r"C:\Users\User\Projects\FYP2\model_enhancement_results\enhanced_v2\checkpoints\v2_epoch=02_val_recall=0.978_val_f1=0.529.ckpt"
    source_path = r"C:\Users\User\Projects\FYP2\Training_Sample_5pct.zip"

    results = evaluate_model(
        checkpoint_path=checkpoint_path,
        source_path=source_path,
        batch_size=32,
    )

    # Save results
    output_path = project_root / "model_enhancement_results" / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
