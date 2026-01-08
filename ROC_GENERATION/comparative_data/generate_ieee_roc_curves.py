"""
Generate IEEE-style ROC Comparison Figure.
Uses confusion matrix data to show operating points in ROC space.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
STATS_FILE = BASE_DIR / "plots_v1" / "statistical_analysis_v1.json"
OUTPUT_DIR = BASE_DIR.parent.parent / "docs" / "figures"


def load_stats():
    with open(STATS_FILE, 'r') as f:
        return json.load(f)


def plot_ieee_roc_space(stats, output_path):
    """
    Create IEEE-style ROC space plot showing operating points.
    Since we have confusion matrices (not probability predictions),
    we show the operating points where each model operates.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # IEEE single column

    # Extract data
    cent_runs = stats['fairness_metrics']['centralized']['per_run']
    fed_runs = stats['fairness_metrics']['federated']['per_run']
    cent_auroc = stats['summary_stats']['centralized']['auroc']
    fed_auroc = stats['summary_stats']['federated']['auroc']

    # Compute points
    cent_fpr = np.array([r['fpr'] for r in cent_runs])
    cent_tpr = np.array([r['sensitivity'] for r in cent_runs])
    fed_fpr = np.array([r['fpr'] for r in fed_runs])
    fed_tpr = np.array([r['sensitivity'] for r in fed_runs])

    # Mean and std
    cent_mean = (np.mean(cent_fpr), np.mean(cent_tpr))
    cent_std = (np.std(cent_fpr), np.std(cent_tpr))
    fed_mean = (np.mean(fed_fpr), np.mean(fed_tpr))
    fed_std = (np.std(fed_fpr), np.std(fed_tpr))

    # Plot individual points (smaller, lighter)
    ax.scatter(cent_fpr, cent_tpr, marker='o', s=30, facecolors='white',
               edgecolors='black', linewidths=0.8, alpha=0.7, zorder=2)
    ax.scatter(fed_fpr, fed_tpr, marker='s', s=30, facecolors='white',
               edgecolors='gray', linewidths=0.8, alpha=0.7, zorder=2)

    # Plot mean with error bars
    ax.errorbar(cent_mean[0], cent_mean[1], xerr=cent_std[0], yerr=cent_std[1],
                fmt='o', color='black', markersize=8, capsize=4, capthick=1.2,
                elinewidth=1.2, markerfacecolor='black', markeredgewidth=1.2,
                label=f'Centralized (AUC={cent_auroc["mean"]:.3f}±{cent_auroc["std"]:.3f})',
                zorder=3)

    ax.errorbar(fed_mean[0], fed_mean[1], xerr=fed_std[0], yerr=fed_std[1],
                fmt='s', color='gray', markersize=8, capsize=4, capthick=1.2,
                elinewidth=1.2, markerfacecolor='gray', markeredgewidth=1.2,
                label=f'Federated (AUC={fed_auroc["mean"]:.3f}±{fed_auroc["std"]:.3f})',
                zorder=3)

    # Connect means with dotted line to show trade-off
    ax.plot([cent_mean[0], fed_mean[0]], [cent_mean[1], fed_mean[1]],
            'k:', linewidth=0.8, alpha=0.5)

    # Diagonal reference (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.6, alpha=0.4)

    # Add annotation for "better" direction
    ax.annotate('', xy=(0.1, 0.9), xytext=(0.25, 0.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax.text(0.08, 0.85, 'Better', fontsize=8, color='gray', style='italic')

    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.82, 1.02])  # Zoom in on relevant region
    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='none')

    # Grid
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)

    # Tick formatting
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ieee_full_roc_space(stats, output_path):
    """
    Full ROC space view (0 to 1 on both axes).
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    cent_runs = stats['fairness_metrics']['centralized']['per_run']
    fed_runs = stats['fairness_metrics']['federated']['per_run']
    cent_auroc = stats['summary_stats']['centralized']['auroc']
    fed_auroc = stats['summary_stats']['federated']['auroc']

    cent_fpr = np.array([r['fpr'] for r in cent_runs])
    cent_tpr = np.array([r['sensitivity'] for r in cent_runs])
    fed_fpr = np.array([r['fpr'] for r in fed_runs])
    fed_tpr = np.array([r['sensitivity'] for r in fed_runs])

    cent_mean = (np.mean(cent_fpr), np.mean(cent_tpr))
    cent_std = (np.std(cent_fpr), np.std(cent_tpr))
    fed_mean = (np.mean(fed_fpr), np.mean(fed_tpr))
    fed_std = (np.std(fed_fpr), np.std(fed_tpr))

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.6, alpha=0.4, label='Random')

    # Individual points
    ax.scatter(cent_fpr, cent_tpr, marker='o', s=40, facecolors='white',
               edgecolors='black', linewidths=1, alpha=0.8, zorder=2)
    ax.scatter(fed_fpr, fed_tpr, marker='s', s=40, facecolors='white',
               edgecolors='gray', linewidths=1, alpha=0.8, zorder=2)

    # Mean with error bars
    ax.errorbar(cent_mean[0], cent_mean[1], xerr=cent_std[0], yerr=cent_std[1],
                fmt='o', color='black', markersize=10, capsize=5, capthick=1.5,
                elinewidth=1.5, markerfacecolor='black',
                label=f'Centralized\n(AUC={cent_auroc["mean"]:.3f}±{cent_auroc["std"]:.3f})',
                zorder=3)

    ax.errorbar(fed_mean[0], fed_mean[1], xerr=fed_std[0], yerr=fed_std[1],
                fmt='s', color='gray', markersize=10, capsize=5, capthick=1.5,
                elinewidth=1.5, markerfacecolor='gray',
                label=f'Federated\n(AUC={fed_auroc["mean"]:.3f}±{fed_auroc["std"]:.3f})',
                zorder=3)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_aspect('equal')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading statistics...")
    stats = load_stats()

    print("\nGenerating IEEE-style ROC figures...")

    # Zoomed version (better for seeing differences)
    plot_ieee_roc_space(stats, OUTPUT_DIR / "fig_6_11_roc_curves.png")

    # Full view version
    plot_ieee_full_roc_space(stats, OUTPUT_DIR / "fig_6_11_roc_curves_full.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
