"""
Generate ROC Curves and Training Dynamics plots for Chapter 6.
Figures 6.11 and 6.12 for FYP2 thesis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
STATS_FILE = BASE_DIR / "plots_v1" / "statistical_analysis_v1.json"
CENTRALIZED_DIR = BASE_DIR.parent / "results" / "CentralizedV1" / "centralized"
FEDERATED_DIR = BASE_DIR / "federated"
OUTPUT_DIR = BASE_DIR.parent.parent / "docs" / "figures"

# Style
plt.style.use('seaborn-v0_8-whitegrid')
CENTRALIZED_COLOR = '#2ecc71'
FEDERATED_COLOR = '#3498db'
DPI = 150


def load_stats():
    """Load statistical analysis data."""
    with open(STATS_FILE, 'r') as f:
        return json.load(f)


def plot_roc_curves(stats: dict, output_path: Path):
    """
    Plot ROC operating points for centralized and federated.
    Uses confusion matrix data to compute (FPR, TPR) per run.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract per-run data
    cent_runs = stats['fairness_metrics']['centralized']['per_run']
    fed_runs = stats['fairness_metrics']['federated']['per_run']

    # Compute FPR, TPR for each run
    cent_fpr = [r['fpr'] for r in cent_runs]
    cent_tpr = [r['sensitivity'] for r in cent_runs]
    fed_fpr = [r['fpr'] for r in fed_runs]
    fed_tpr = [r['sensitivity'] for r in fed_runs]

    # Plot individual points
    ax.scatter(cent_fpr, cent_tpr, c=CENTRALIZED_COLOR, s=100, alpha=0.7,
               label='Centralized runs', marker='o', edgecolors='white', linewidths=1.5)
    ax.scatter(fed_fpr, fed_tpr, c=FEDERATED_COLOR, s=100, alpha=0.7,
               label='Federated runs', marker='s', edgecolors='white', linewidths=1.5)

    # Plot mean points with larger markers
    cent_mean_fpr, cent_mean_tpr = np.mean(cent_fpr), np.mean(cent_tpr)
    fed_mean_fpr, fed_mean_tpr = np.mean(fed_fpr), np.mean(fed_tpr)

    ax.scatter([cent_mean_fpr], [cent_mean_tpr], c=CENTRALIZED_COLOR, s=250,
               marker='o', edgecolors='black', linewidths=2, zorder=5)
    ax.scatter([fed_mean_fpr], [fed_mean_tpr], c=FEDERATED_COLOR, s=250,
               marker='s', edgecolors='black', linewidths=2, zorder=5)

    # Add ellipses showing ±1 SD spread
    from matplotlib.patches import Ellipse
    cent_std_fpr, cent_std_tpr = np.std(cent_fpr), np.std(cent_tpr)
    fed_std_fpr, fed_std_tpr = np.std(fed_fpr), np.std(fed_tpr)

    ellipse_cent = Ellipse((cent_mean_fpr, cent_mean_tpr),
                           width=2*cent_std_fpr, height=2*cent_std_tpr,
                           facecolor=CENTRALIZED_COLOR, alpha=0.2, edgecolor=CENTRALIZED_COLOR)
    ellipse_fed = Ellipse((fed_mean_fpr, fed_mean_tpr),
                          width=2*fed_std_fpr, height=2*fed_std_tpr,
                          facecolor=FEDERATED_COLOR, alpha=0.2, edgecolor=FEDERATED_COLOR)
    ax.add_patch(ellipse_cent)
    ax.add_patch(ellipse_fed)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random classifier')

    # Annotations with AUROC
    cent_auroc = stats['summary_stats']['centralized']['auroc']
    fed_auroc = stats['summary_stats']['federated']['auroc']

    ax.annotate(f"Centralized\nAUC={cent_auroc['mean']:.3f}±{cent_auroc['std']:.3f}",
                xy=(cent_mean_fpr, cent_mean_tpr), xytext=(cent_mean_fpr + 0.15, cent_mean_tpr - 0.1),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color=CENTRALIZED_COLOR, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=CENTRALIZED_COLOR))

    ax.annotate(f"Federated\nAUC={fed_auroc['mean']:.3f}±{fed_auroc['std']:.3f}",
                xy=(fed_mean_fpr, fed_mean_tpr), xytext=(fed_mean_fpr + 0.1, fed_mean_tpr + 0.05),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color=FEDERATED_COLOR, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=FEDERATED_COLOR))

    # Labels and legend
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Operating Points: Centralized vs Federated Learning', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_dynamics(output_path: Path):
    """
    Plot training dynamics showing loss curves.
    Subplot (a): Centralized - per-epoch loss
    Subplot (b): Federated - per-round loss
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Centralized - load CSVs
    ax_cent = axes[0]
    cent_losses = []

    for run_id in range(5):
        csv_path = CENTRALIZED_DIR / f"run_{run_id}" / "logs" / "metrics" / f"centralized_run_{run_id}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if 'val_loss' in df.columns:
                cent_losses.append(df['val_loss'].values)
                ax_cent.plot(df['epoch'], df['val_loss'], alpha=0.4, color=CENTRALIZED_COLOR, linewidth=1)

    # Mean and SD
    if cent_losses:
        max_len = max(len(l) for l in cent_losses)
        padded = np.array([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in cent_losses])
        mean_loss = np.nanmean(padded, axis=0)
        std_loss = np.nanstd(padded, axis=0)
        epochs = np.arange(max_len)

        ax_cent.plot(epochs, mean_loss, color=CENTRALIZED_COLOR, linewidth=2.5, label='Mean')
        ax_cent.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                             color=CENTRALIZED_COLOR, alpha=0.2, label='±1 SD')

    ax_cent.set_xlabel('Epoch', fontsize=11)
    ax_cent.set_ylabel('Validation Loss', fontsize=11)
    ax_cent.set_title('(a) Centralized Training', fontsize=12, fontweight='bold')
    ax_cent.legend(loc='upper right')
    ax_cent.grid(True, alpha=0.3)

    # (b) Federated - load JSONs
    ax_fed = axes[1]
    fed_losses = []

    for run_id in range(7):  # results_analysis_0.json to results_analysis_6.json
        json_path = FEDERATED_DIR / f"results_analysis_{run_id}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            # Parse evaluate_metrics_serverapp - keys are round numbers as strings
            if 'evaluate_metrics_serverapp' in data:
                server_metrics = data['evaluate_metrics_serverapp']
                losses = []
                for round_key in sorted(server_metrics.keys(), key=int):
                    # Values are string representations of dicts
                    metric_str = server_metrics[round_key]
                    if isinstance(metric_str, str):
                        # Parse the string as a dict using eval (safe here - our own data)
                        import ast
                        metric_dict = ast.literal_eval(metric_str)
                    else:
                        metric_dict = metric_str
                    if 'server_loss' in metric_dict:
                        losses.append(metric_dict['server_loss'])
                if losses:
                    fed_losses.append(losses)
                    rounds = list(range(len(losses)))
                    ax_fed.plot(rounds, losses, alpha=0.4, color=FEDERATED_COLOR, linewidth=1)

    # Mean and SD
    if fed_losses:
        max_len = max(len(l) for l in fed_losses)
        padded = np.array([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in fed_losses])
        mean_loss = np.nanmean(padded, axis=0)
        std_loss = np.nanstd(padded, axis=0)
        rounds = np.arange(max_len)

        ax_fed.plot(rounds, mean_loss, color=FEDERATED_COLOR, linewidth=2.5, label='Mean')
        ax_fed.fill_between(rounds, mean_loss - std_loss, mean_loss + std_loss,
                            color=FEDERATED_COLOR, alpha=0.2, label='±1 SD')

        # Add vertical lines for aggregation rounds
        for r in rounds[:-1]:
            ax_fed.axvline(x=r + 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax_fed.set_xlabel('Communication Round', fontsize=11)
    ax_fed.set_ylabel('Validation Loss', fontsize=11)
    ax_fed.set_title('(b) Federated Training', fontsize=12, fontweight='bold')
    ax_fed.legend(loc='upper right')
    ax_fed.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading statistics...")
    stats = load_stats()

    print("\nGenerating Fig 6.11: ROC Curves...")
    plot_roc_curves(stats, OUTPUT_DIR / "fig_6_11_roc_curves.png")

    print("\nGenerating Fig 6.12: Training Dynamics...")
    plot_training_dynamics(OUTPUT_DIR / "fig_6_12_training_dynamics.png")

    print("\nDone! Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
