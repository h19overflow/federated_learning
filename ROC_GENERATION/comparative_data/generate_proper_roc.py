"""
Generate Proper ROC Curves from Checkpoint Inference.
Runs inference on validation data to get probability predictions,
then generates actual ROC curves (not just operating points).
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from torchvision import transforms
from tqdm import tqdm

# IEEE style settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['axes.linewidth'] = 0.8

# Colors
CENTRALIZED_COLOR = '#2ecc71'  # Green
FEDERATED_COLOR = '#3498db'    # Blue

# Paths
BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR.parent / "results" / "CentralizedV1" / "centralized"
DATA_DIR = BASE_DIR.parent / "extracted_data"
TRAIN_METADATA = DATA_DIR / "stage2_train_metadata.csv"
IMAGE_DIR = DATA_DIR / "Training" / "Images"
OUTPUT_DIR = BASE_DIR.parent.parent / "docs" / "figures"
STATS_FILE = BASE_DIR / "plots_v1" / "statistical_analysis_v1.json"

# Checkpoint mappings (run_id -> best checkpoint filename)
CENTRALIZED_CHECKPOINTS = {
    0: "pneumonia_model_09_0.927.ckpt",
    1: "pneumonia_model_06_0.915.ckpt",
    2: "pneumonia_model_09_0.958.ckpt",
    3: "pneumonia_model_00_0.980.ckpt",
    4: "pneumonia_model_07_0.928.ckpt",
}

CENTRALIZED_SEEDS = {0: 42, 1: 43, 2: 44, 3: 45, 4: 46}


def get_val_transforms():
    """Get validation transforms (must match training)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def prepare_dataframe(df, image_dir):
    """Prepare dataframe with filename column, filtering to available images."""
    # Deduplicate by patientId (keep first row for each patient)
    df_unique = df.drop_duplicates(subset='patientId', keep='first').reset_index(drop=True)

    # Add filename column
    df_unique['filename'] = df_unique['patientId'].astype(str) + '.png'

    # Filter to only images that exist
    available_images = set(f.name for f in Path(image_dir).glob('*.png'))
    df_filtered = df_unique[df_unique['filename'].isin(available_images)].reset_index(drop=True)

    print(f"Filtered to {len(df_filtered)} patients with available images (from {len(df_unique)})")

    return df_filtered


def load_model_and_predict(checkpoint_path, val_loader, device):
    """Load checkpoint and run inference to get probabilities."""
    from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet_enhanced import LitResNetEnhanced

    print(f"Loading checkpoint: {checkpoint_path.name}")

    # Load model
    model = LitResNetEnhanced.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Running inference", leave=False):
            images, labels = batch
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    return np.array(all_probs), np.array(all_labels)


def compute_roc_curves_centralized(device='cpu'):
    """Compute ROC curves for all centralized runs."""
    from federated_pneumonia_detection.src.entities.custom_image_dataset import CustomImageDataset

    print("Loading training metadata...")
    df = pd.read_csv(TRAIN_METADATA)
    df = prepare_dataframe(df, IMAGE_DIR)

    print(f"Total available patients: {len(df)}")
    print(f"Class distribution: {df['Target'].value_counts().to_dict()}")

    val_transforms = get_val_transforms()

    all_roc_data = []

    for run_id, ckpt_name in CENTRALIZED_CHECKPOINTS.items():
        seed = CENTRALIZED_SEEDS[run_id]
        print(f"\n=== Run {run_id} (seed={seed}) ===")

        # Split with same seed as training
        _, val_df = train_test_split(
            df, test_size=0.2, stratify=df['Target'], random_state=seed
        )
        val_df = val_df.reset_index(drop=True)

        print(f"Validation samples: {len(val_df)}")

        # Create dataset and loader
        val_dataset = CustomImageDataset(
            val_df, IMAGE_DIR, transform=val_transforms,
            filename_column='filename', target_column='Target',
            validate_images=False
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        # Load checkpoint and predict
        ckpt_path = CHECKPOINT_DIR / f"run_{run_id}" / "checkpoints" / ckpt_name

        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue

        probs, labels = load_model_and_predict(ckpt_path, val_loader, device)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        print(f"AUC: {roc_auc:.4f}")

        all_roc_data.append({
            'run_id': run_id,
            'seed': seed,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        })

    return all_roc_data


def load_stats_from_paper():
    """Load operating points and AUROC values from statistical analysis (paper values)."""
    with open(STATS_FILE, 'r') as f:
        stats = json.load(f)

    fed_runs = stats['fairness_metrics']['federated']['per_run']
    fed_auroc = stats['summary_stats']['federated']['auroc']
    cent_auroc = stats['summary_stats']['centralized']['auroc']

    points = []
    for run in fed_runs:
        points.append({
            'fpr': run['fpr'],
            'tpr': run['sensitivity'],
        })

    return points, fed_auroc, cent_auroc


def interpolate_roc_curves(all_roc_data, n_points=101):
    """Interpolate ROC curves to common FPR points for averaging."""
    common_fpr = np.linspace(0, 1, n_points)

    interpolated_tprs = []
    for data in all_roc_data:
        interp_tpr = np.interp(common_fpr, data['fpr'], data['tpr'])
        interp_tpr[0] = 0.0  # Start at (0, 0)
        interpolated_tprs.append(interp_tpr)

    return common_fpr, np.array(interpolated_tprs)


def plot_ieee_roc(centralized_data, federated_points, fed_auroc, cent_auroc, output_path):
    """Create IEEE-style ROC curve figure with colors."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Interpolate centralized curves
    common_fpr, tpr_matrix = interpolate_roc_curves(centralized_data)
    mean_tpr = np.mean(tpr_matrix, axis=0)
    std_tpr = np.std(tpr_matrix, axis=0)

    # Use official paper AUC values for legend
    cent_auc_mean = cent_auroc['mean']
    cent_auc_std = cent_auroc['std']

    # Plot individual centralized curves (faint)
    for data in centralized_data:
        ax.plot(data['fpr'], data['tpr'], color=CENTRALIZED_COLOR, alpha=0.2, linewidth=0.5)

    # Plot mean centralized curve with shading
    ax.plot(common_fpr, mean_tpr, color=CENTRALIZED_COLOR, linewidth=2,
            label=f'Centralized (AUC={cent_auc_mean:.3f}±{cent_auc_std:.3f})')
    ax.fill_between(common_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color=CENTRALIZED_COLOR, alpha=0.2)

    # Plot federated operating points
    fed_fprs = [p['fpr'] for p in federated_points]
    fed_tprs = [p['tpr'] for p in federated_points]

    # Individual points
    ax.scatter(fed_fprs, fed_tprs, marker='s', s=50, facecolors='white',
               edgecolors=FEDERATED_COLOR, linewidths=1.5, alpha=0.9, zorder=4)

    # Mean point with error bars
    fed_mean_fpr = np.mean(fed_fprs)
    fed_mean_tpr = np.mean(fed_tprs)
    fed_std_fpr = np.std(fed_fprs)
    fed_std_tpr = np.std(fed_tprs)

    ax.errorbar(fed_mean_fpr, fed_mean_tpr, xerr=fed_std_fpr, yerr=fed_std_tpr,
                fmt='s', color=FEDERATED_COLOR, markersize=10, capsize=4, capthick=1.5,
                elinewidth=1.5, markerfacecolor=FEDERATED_COLOR,
                label=f'Federated (AUC={fed_auroc["mean"]:.3f}±{fed_auroc["std"]:.3f})',
                zorder=5)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random')

    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_aspect('equal')
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='none')
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ieee_roc_zoomed(centralized_data, federated_points, fed_auroc, cent_auroc, output_path):
    """Create zoomed IEEE-style ROC curve for high-performance region with colors."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Interpolate centralized curves
    common_fpr, tpr_matrix = interpolate_roc_curves(centralized_data)
    mean_tpr = np.mean(tpr_matrix, axis=0)
    std_tpr = np.std(tpr_matrix, axis=0)

    # Use official paper AUC values for legend
    cent_auc_mean = cent_auroc['mean']
    cent_auc_std = cent_auroc['std']

    # Plot individual centralized curves (faint)
    for data in centralized_data:
        ax.plot(data['fpr'], data['tpr'], color=CENTRALIZED_COLOR, alpha=0.2, linewidth=0.5)

    # Plot mean with shading
    ax.plot(common_fpr, mean_tpr, color=CENTRALIZED_COLOR, linewidth=2,
            label=f'Centralized\n(AUC={cent_auc_mean:.3f}±{cent_auc_std:.3f})')
    ax.fill_between(common_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color=CENTRALIZED_COLOR, alpha=0.2)

    # Federated points
    fed_fprs = [p['fpr'] for p in federated_points]
    fed_tprs = [p['tpr'] for p in federated_points]

    ax.scatter(fed_fprs, fed_tprs, marker='s', s=50, facecolors='white',
               edgecolors=FEDERATED_COLOR, linewidths=1.5, alpha=0.9, zorder=4)

    fed_mean_fpr = np.mean(fed_fprs)
    fed_mean_tpr = np.mean(fed_tprs)
    fed_std_fpr = np.std(fed_fprs)
    fed_std_tpr = np.std(fed_tprs)

    ax.errorbar(fed_mean_fpr, fed_mean_tpr, xerr=fed_std_fpr, yerr=fed_std_tpr,
                fmt='s', color=FEDERATED_COLOR, markersize=10, capsize=4, capthick=1.5,
                elinewidth=1.5, markerfacecolor=FEDERATED_COLOR,
                label=f'Federated\n(AUC={fed_auroc["mean"]:.3f}±{fed_auroc["std"]:.3f})',
                zorder=5)

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)

    # Zoomed view - focus on top-left region
    ax.set_xlim([-0.02, 0.6])
    ax.set_ylim([0.75, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='none')
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Compute centralized ROC curves via inference
    print("\n" + "="*50)
    print("Computing Centralized ROC Curves from Inference")
    print("="*50)
    centralized_data = compute_roc_curves_centralized(device)

    if not centralized_data:
        print("ERROR: No centralized data computed. Check checkpoint paths.")
        return

    # Load paper values (official AUROC + federated operating points)
    print("\n" + "="*50)
    print("Loading Paper Values (Official AUROC)")
    print("="*50)
    federated_points, fed_auroc, cent_auroc = load_stats_from_paper()
    print(f"Loaded {len(federated_points)} federated operating points")
    print(f"Centralized AUC (paper): {cent_auroc['mean']:.3f}±{cent_auroc['std']:.3f}")
    print(f"Federated AUC (paper): {fed_auroc['mean']:.3f}±{fed_auroc['std']:.3f}")

    # Generate plots
    print("\n" + "="*50)
    print("Generating IEEE-style ROC Figures")
    print("="*50)

    plot_ieee_roc(centralized_data, federated_points, fed_auroc, cent_auroc,
                  OUTPUT_DIR / "fig_6_11_roc_curves.png")

    plot_ieee_roc_zoomed(centralized_data, federated_points, fed_auroc, cent_auroc,
                         OUTPUT_DIR / "fig_6_11_roc_curves_zoomed.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
