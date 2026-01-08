"""
Evaluation script for comparing centralized vs federated pneumonia detection models.

Loads both best checkpoints, runs inference on test data, and generates:
- Confusion matrices
- Metrics: accuracy, precision, recall, F1, AUROC
- Comparison visualizations
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from tqdm import tqdm

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
)

# =============================================================================
# STYLE CONFIGURATION (matching comprehensive_plots_v1.py)
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300  # Research paper quality

# Colors (matching FYP2 style)
CENT_COLOR = '#2ecc71'    # Green
FED_COLOR = '#3498db'     # Blue
BASELINE_COLOR = '#e74c3c'  # Red
HIGHLIGHT = '#f39c12'     # Orange


class FolderImageDataset(Dataset):
    """Dataset that loads images from NORMAL/ and PNEUMONIA/ folders."""

    def __init__(self, root_dir: Path, transform=None):
        """
        Initialize dataset from folder structure.

        Args:
            root_dir: Root directory containing NORMAL/ and PNEUMONIA/ folders
            transform: Transform pipeline to apply
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Load from NORMAL folder (label=0)
        normal_dir = self.root_dir / "NORMAL"
        if normal_dir.exists():
            for img_path in normal_dir.glob("*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append((img_path, 0))

        # Load from PNEUMONIA folder (label=1)
        pneumonia_dir = self.root_dir / "PNEUMONIA"
        if pneumonia_dir.exists():
            for img_path in pneumonia_dir.glob("*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append((img_path, 1))

        print(f"Loaded {len(self.samples)} images from {root_dir}")
        print(
            f"  - NORMAL: {sum(1 for _, label in self.samples if label == 0)} images"
        )
        print(
            f"  - PNEUMONIA: {sum(1 for _, label in self.samples if label == 1)} images"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def get_test_transforms(img_size=(224, 224)):
    """
    Create test/validation transform pipeline matching training.

    Args:
        img_size: Target image size

    Returns:
        Transform pipeline
    """
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """
    Load LitResNetEnhanced model from checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    print(f"\nLoading checkpoint: {checkpoint_path.name}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    model = LitResNetEnhanced.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    print(f"  ✓ Model loaded successfully")
    return model


@torch.no_grad()
def evaluate_model(
    model, dataloader: DataLoader, device: torch.device, model_name: str
) -> Dict:
    """
    Evaluate model and compute all metrics.

    Args:
        model: PyTorch Lightning model
        dataloader: Test data loader
        device: Device to run on
        model_name: Name for display

    Returns:
        Dictionary with all metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name}")
    print(f"{'=' * 60}")

    all_preds = []
    all_probs = []
    all_labels = []

    # Run inference
    for images, labels in tqdm(dataloader, desc="Inference"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze()

        # Store results
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert to numpy
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)
    all_preds = (all_probs >= 0.5).astype(int)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "auroc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
    }

    # Extract confusion matrix components
    tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    # Compute additional metrics
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["sensitivity"] = metrics["recall"]  # Same as recall
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    print(f"  AUROC:      {metrics['auroc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TP: {tp:4d}")
    print(f"\nClinical Metrics:")
    print(f"  Sensitivity (TPR): {metrics['sensitivity']:.4f}")
    print(f"  Specificity (TNR): {metrics['specificity']:.4f}")
    print(f"  False Negative Rate: {metrics['fnr']:.4f}")
    print(f"  False Positive Rate: {metrics['fpr']:.4f}")

    return metrics


def plot_confusion_matrices(
    centralized_metrics: Dict, federated_metrics: Dict, output_dir: Path
):
    """
    Plot side-by-side confusion matrices (matching comprehensive_plots_v1.py style).

    Args:
        centralized_metrics: Metrics from centralized model
        federated_metrics: Metrics from federated model
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (paradigm, metrics, color) in enumerate([
        ('Centralized', centralized_metrics, 'Greens'),
        ('Federated', federated_metrics, 'Blues')
    ]):
        cm = metrics["confusion_matrix"]
        tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]

        # Normalize to percentages
        total = cm.sum()
        cm_norm = cm / total * 100

        # Plot heatmap without default annotations
        sns.heatmap(
            cm_norm,
            annot=False,
            cmap=color,
            ax=axes[idx],
            cbar=True,
            vmin=0,
            vmax=60,
            xticklabels=['Predicted\nNormal', 'Predicted\nPneumonia'],
            yticklabels=['Actual\nNormal', 'Actual\nPneumonia']
        )

        # Add custom annotations with counts and percentages
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_norm[i, j]
                text_color = 'white' if pct > 30 else 'black'
                axes[idx].annotate(
                    f'{count:,.0f}\n({pct:.1f}%)',
                    xy=(j + 0.5, i + 0.5),
                    ha='center',
                    va='center',
                    fontsize=12,
                    fontweight='bold',
                    color=text_color
                )

        axes[idx].set_title(
            f'{paradigm} Model\nRecall: {metrics["recall"]:.3f}',
            fontsize=13
        )

    plt.suptitle('Confusion Matrices: Test Set Evaluation', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved confusion matrices to {output_dir / 'confusion_matrices_comparison.png'}")
    plt.close()


def plot_roc_curves(
    centralized_metrics: Dict, federated_metrics: Dict, output_dir: Path
):
    """
    Plot ROC curves for both models (matching FYP2 style).

    Args:
        centralized_metrics: Metrics from centralized model
        federated_metrics: Metrics from federated model
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute ROC curves
    fpr_cent, tpr_cent, _ = roc_curve(
        centralized_metrics["labels"], centralized_metrics["probabilities"]
    )
    fpr_fed, tpr_fed, _ = roc_curve(
        federated_metrics["labels"], federated_metrics["probabilities"]
    )

    # Plot ROC curves with FYP2 colors
    ax.plot(
        fpr_cent,
        tpr_cent,
        label=f"Centralized (AUROC = {centralized_metrics['auroc']:.3f})",
        linewidth=2.5,
        color=CENT_COLOR,
        alpha=0.9
    )
    ax.plot(
        fpr_fed,
        tpr_fed,
        label=f"Federated (AUROC = {federated_metrics['auroc']:.3f})",
        linewidth=2.5,
        color=FED_COLOR,
        alpha=0.9
    )

    # Reference diagonal
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve Comparison: Test Set Evaluation", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved ROC curves to {output_dir / 'roc_curves_comparison.png'}")
    plt.close()


def plot_metrics_comparison(
    centralized_metrics: Dict, federated_metrics: Dict, output_dir: Path
):
    """
    Plot bar chart comparing all metrics (matching comprehensive_plots_v1.py style).

    Args:
        centralized_metrics: Metrics from centralized model
        federated_metrics: Metrics from federated model
        output_dir: Directory to save plots
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "auroc"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUROC"]

    cent_values = [centralized_metrics[m] for m in metrics_to_plot]
    fed_values = [federated_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    # Use FYP2 colors with alpha and edge
    bars1 = ax.bar(
        x - width / 2, cent_values, width,
        label="Centralized",
        color=CENT_COLOR,
        alpha=0.85,
        edgecolor='black',
        linewidth=1
    )
    bars2 = ax.bar(
        x + width / 2, fed_values, width,
        label="Federated",
        color=FED_COLOR,
        alpha=0.85,
        edgecolor='black',
        linewidth=1
    )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Metrics: Test Set Evaluation\nCentralized vs Federated", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.15)

    # Add clinical threshold
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.annotate('Clinical threshold (0.85)', xy=(4.5, 0.86), fontsize=9, color='gray', alpha=0.7)

    # Add value labels on bars
    for bar, val in zip(bars1, cent_values):
        ax.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, val + 0.02),
            ha='center',
            fontsize=9,
            fontweight='bold'
        )
    for bar, val in zip(bars2, fed_values):
        ax.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, val + 0.02),
            ha='center',
            fontsize=9,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved metrics comparison to {output_dir / 'metrics_comparison.png'}")
    plt.close()


def save_results_to_files(
    centralized_metrics: Dict,
    federated_metrics: Dict,
    output_dir: Path,
    test_data_path: Path,
):
    """
    Save all results to JSON and CSV files.

    Args:
        centralized_metrics: Metrics from centralized model
        federated_metrics: Metrics from federated model
        output_dir: Directory to save results
        test_data_path: Path to test data
    """
    # Prepare results dictionary
    results = {
        "test_data_path": str(test_data_path),
        "num_samples": len(centralized_metrics["labels"]),
        "class_distribution": {
            "normal": int(np.sum(centralized_metrics["labels"] == 0)),
            "pneumonia": int(np.sum(centralized_metrics["labels"] == 1)),
        },
        "centralized": {
            "checkpoint": "pneumonia_model_00_0.980.ckpt",
            "accuracy": float(centralized_metrics["accuracy"]),
            "precision": float(centralized_metrics["precision"]),
            "recall": float(centralized_metrics["recall"]),
            "f1_score": float(centralized_metrics["f1_score"]),
            "auroc": float(centralized_metrics["auroc"]),
            "confusion_matrix": {
                "tn": centralized_metrics["tn"],
                "fp": centralized_metrics["fp"],
                "fn": centralized_metrics["fn"],
                "tp": centralized_metrics["tp"],
            },
            "clinical_metrics": {
                "sensitivity": float(centralized_metrics["sensitivity"]),
                "specificity": float(centralized_metrics["specificity"]),
                "fnr": float(centralized_metrics["fnr"]),
                "fpr": float(centralized_metrics["fpr"]),
            },
        },
        "federated": {
            "checkpoint": "pneumonia_model_01_0.988.ckpt",
            "accuracy": float(federated_metrics["accuracy"]),
            "precision": float(federated_metrics["precision"]),
            "recall": float(federated_metrics["recall"]),
            "f1_score": float(federated_metrics["f1_score"]),
            "auroc": float(federated_metrics["auroc"]),
            "confusion_matrix": {
                "tn": federated_metrics["tn"],
                "fp": federated_metrics["fp"],
                "fn": federated_metrics["fn"],
                "tp": federated_metrics["tp"],
            },
            "clinical_metrics": {
                "sensitivity": float(federated_metrics["sensitivity"]),
                "specificity": float(federated_metrics["specificity"]),
                "fnr": float(federated_metrics["fnr"]),
                "fpr": float(federated_metrics["fpr"]),
            },
        },
    }

    # Save JSON
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {json_path}")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "AUROC",
                "Sensitivity",
                "Specificity",
                "FNR",
                "FPR",
            ],
            "Centralized": [
                centralized_metrics["accuracy"],
                centralized_metrics["precision"],
                centralized_metrics["recall"],
                centralized_metrics["f1_score"],
                centralized_metrics["auroc"],
                centralized_metrics["sensitivity"],
                centralized_metrics["specificity"],
                centralized_metrics["fnr"],
                centralized_metrics["fpr"],
            ],
            "Federated": [
                federated_metrics["accuracy"],
                federated_metrics["precision"],
                federated_metrics["recall"],
                federated_metrics["f1_score"],
                federated_metrics["auroc"],
                federated_metrics["sensitivity"],
                federated_metrics["specificity"],
                federated_metrics["fnr"],
                federated_metrics["fpr"],
            ],
        }
    )

    # Add difference column
    comparison_df["Difference (Fed - Cent)"] = (
        comparison_df["Federated"] - comparison_df["Centralized"]
    )

    # Save CSV
    csv_path = output_dir / "metrics_comparison.csv"
    comparison_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"✓ Saved comparison table to {csv_path}")

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("METRICS COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(comparison_df.to_string(index=False))
    print(f"{'=' * 80}")


def main():
    """Main evaluation pipeline."""
    warnings.filterwarnings("ignore")

    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent
    ROC_GEN_DIR = PROJECT_ROOT / "ROC_GENERATION"

    # Checkpoint paths (best models from earlier analysis)
    CENTRALIZED_CKPT = (
        ROC_GEN_DIR
        / "results/CentralizedV1/centralized/run_3/checkpoints/pneumonia_model_00_0.980.ckpt"
    )
    FEDERATED_CKPT = ROC_GEN_DIR / "results/checkpoints/pneumonia_model_01_0.988.ckpt"

    # Output directory
    OUTPUT_DIR = ROC_GEN_DIR / "evaluation_results"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("PNEUMONIA DETECTION MODEL EVALUATION")
    print("Comparing Centralized vs Federated Learning")
    print("=" * 80)

    # Get test data path from user
    print("\nPlease provide the path to your test data directory.")
    print("Expected structure:")
    print("  test_data/")
    print("    ├── NORMAL/")
    print("    │   ├── image1.png")
    print("    │   └── ...")
    print("    └── PNEUMONIA/")
    print("        ├── image1.png")
    print("        └── ...")

    test_data_path = input("\nTest data path: ").strip().strip('"').strip("'")
    test_data_path = Path(test_data_path)

    if not test_data_path.exists():
        print(f"\n❌ Error: Path does not exist: {test_data_path}")
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create dataset and dataloader
    print("\nLoading test data...")
    test_transform = get_test_transforms(img_size=(224, 224))
    test_dataset = FolderImageDataset(test_data_path, transform=test_transform)

    if len(test_dataset) == 0:
        print("\n❌ Error: No images found in test dataset")
        return

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    # Load models
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    centralized_model = load_model_from_checkpoint(CENTRALIZED_CKPT, device)
    federated_model = load_model_from_checkpoint(FEDERATED_CKPT, device)

    # Evaluate models
    centralized_metrics = evaluate_model(
        centralized_model, test_loader, device, "Centralized Model"
    )
    federated_metrics = evaluate_model(
        federated_model, test_loader, device, "Federated Model"
    )

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_confusion_matrices(centralized_metrics, federated_metrics, OUTPUT_DIR)
    plot_roc_curves(centralized_metrics, federated_metrics, OUTPUT_DIR)
    plot_metrics_comparison(centralized_metrics, federated_metrics, OUTPUT_DIR)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    save_results_to_files(
        centralized_metrics, federated_metrics, OUTPUT_DIR, test_data_path
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - evaluation_results.json (detailed metrics)")
    print("  - metrics_comparison.csv (comparison table)")
    print("  - confusion_matrices_comparison.png")
    print("  - roc_curves_comparison.png")
    print("  - metrics_comparison.png")


if __name__ == "__main__":
    main()
