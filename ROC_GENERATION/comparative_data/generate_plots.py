"""
FYP2 Visualization Script
Generates all charts for Chapter 6: Results & Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Paths
BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(BASE_DIR.parent / "analysis" / "experiment_results.csv")
cent = df[df['paradigm'] == 'centralized']
fed = df[df['paradigm'] == 'federated']

with open(BASE_DIR / "statistical_analysis.json") as f:
    stats = json.load(f)

with open(BASE_DIR / "baseline" / "pre_tuning_federated.json") as f:
    baseline = json.load(f)
baseline_metrics = eval(baseline['evaluate_metrics_serverapp']['2'])

# Colors
CENT_COLOR = '#2ecc71'  # Green
FED_COLOR = '#3498db'   # Blue
BASELINE_COLOR = '#e74c3c'  # Red

print("Generating plots...")

# ============================================================
# PLOT 1: Metric Comparison Bar Chart (Section 6.2.1)
# ============================================================
def plot_metric_comparison():
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']

    cent_means = [stats['summary_stats']['centralized'][m]['mean'] for m in metrics]
    cent_stds = [stats['summary_stats']['centralized'][m]['std'] for m in metrics]
    fed_means = [stats['summary_stats']['federated'][m]['mean'] for m in metrics]
    fed_stds = [stats['summary_stats']['federated'][m]['std'] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, cent_means, width, yerr=cent_stds,
                   label=f'Centralized (n=10)', color=CENT_COLOR, capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width/2, fed_means, width, yerr=fed_stds,
                   label=f'Federated (n=5)', color=FED_COLOR, capsize=5, alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics: Centralized vs Federated Learning')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars1, cent_means):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar, val in zip(bars2, fed_means):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_1_metric_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_1_metric_comparison.png")

# ============================================================
# PLOT 2: Recall Box Plot (Section 6.2.1)
# ============================================================
def plot_recall_boxplot():
    fig, ax = plt.subplots(figsize=(8, 6))

    data = [cent['recall'].values, fed['recall'].values]
    positions = [1, 2]

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(CENT_COLOR)
    bp['boxes'][1].set_facecolor(FED_COLOR)

    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(positions[i], 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, color='black', s=30, zorder=3)

    ax.set_xticklabels(['Centralized\n(n=10)', 'Federated\n(n=5)'])
    ax.set_ylabel('Recall (Sensitivity)')
    ax.set_title('Recall Distribution by Training Paradigm')
    ax.set_ylim(0.7, 1.05)

    # Add mean lines
    ax.axhline(y=cent['recall'].mean(), xmin=0.1, xmax=0.4, color=CENT_COLOR, linestyle='--', alpha=0.7)
    ax.axhline(y=fed['recall'].mean(), xmin=0.6, xmax=0.9, color=FED_COLOR, linestyle='--', alpha=0.7)

    # Add p-value annotation
    p_val = stats['statistical_tests']['recall']['p_value']
    ax.annotate(f'p = {p_val:.4f}*', xy=(1.5, 1.0), fontsize=11, ha='center')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_2_recall_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_2_recall_boxplot.png")

# ============================================================
# PLOT 3: Baseline Comparison (Section 6.2.2)
# ============================================================
def plot_baseline_comparison():
    metrics = ['Recall', 'AUROC', 'F1-Score', 'Accuracy']
    baseline_vals = [
        baseline_metrics['server_recall'],
        baseline_metrics['server_auroc'],
        baseline_metrics['server_f1'],
        baseline_metrics['server_accuracy']
    ]
    cent_vals = [
        cent['recall'].mean(),
        cent['auroc'].mean(),
        cent['f1_score'].mean(),
        cent['accuracy'].mean()
    ]
    fed_vals = [
        fed['recall'].mean(),
        fed['auroc'].mean(),
        fed['f1_score'].mean(),
        fed['accuracy'].mean()
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 7))
    bars1 = ax.bar(x - width, baseline_vals, width, label='Baseline (Dec 26)', color=BASELINE_COLOR, alpha=0.8)
    bars2 = ax.bar(x, cent_vals, width, label='Tuned Centralized', color=CENT_COLOR, alpha=0.8)
    bars3 = ax.bar(x + width, fed_vals, width, label='Tuned Federated', color=FED_COLOR, alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Baseline vs Tuned Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    # Add improvement arrow for recall
    ax.annotate('', xy=(0.25, fed_vals[0]), xytext=(0.25, baseline_vals[0]),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('+24.2%', xy=(0.4, (baseline_vals[0] + fed_vals[0])/2), fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_3_baseline_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_3_baseline_comparison.png")

# ============================================================
# PLOT 4: Coefficient of Variation (Section 6.2.3)
# ============================================================
def plot_cv_comparison():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']

    cent_cv = [stats['coefficient_of_variation']['centralized'][m] for m in metric_keys]
    fed_cv = [stats['coefficient_of_variation']['federated'][m] for m in metric_keys]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cent_cv, width, label='Centralized', color=CENT_COLOR, alpha=0.8)
    bars2 = ax.bar(x + width/2, fed_cv, width, label='Federated', color=FED_COLOR, alpha=0.8)

    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title('Model Stability: Coefficient of Variation by Metric')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add threshold line
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax.annotate('High variability threshold', xy=(4.5, 11), fontsize=9, color='red')

    # Add value labels
    for bar, val in zip(bars1, cent_cv):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar, val in zip(bars2, fed_cv):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_4_cv_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_4_cv_comparison.png")

# ============================================================
# PLOT 5: Statistical Significance (Section 6.2.3)
# ============================================================
def plot_statistical_tests():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']

    p_values = [stats['statistical_tests'][m]['p_value'] for m in metric_keys]
    cohens_d = [abs(stats['statistical_tests'][m]['cohens_d']) for m in metric_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # P-values
    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    bars1 = ax1.barh(metrics, p_values, color=colors, alpha=0.7)
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax1.set_xlabel('p-value')
    ax1.set_title('Statistical Significance (Welch\'s t-test)')
    ax1.legend()
    ax1.set_xlim(0, max(p_values) * 1.2)

    for bar, val in zip(bars1, p_values):
        ax1.annotate(f'{val:.4f}', xy=(val + 0.001, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10)

    # Cohen's d
    effect_colors = ['#27ae60' if d > 0.8 else '#f39c12' if d > 0.5 else '#e74c3c' for d in cohens_d]
    bars2 = ax2.barh(metrics, cohens_d, color=effect_colors, alpha=0.7)
    ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Large effect (0.8)')
    ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium effect (0.5)')
    ax2.set_xlabel('|Cohen\'s d|')
    ax2.set_title('Effect Size (Cohen\'s d)')
    ax2.legend(loc='lower right')

    for bar, val in zip(bars2, cohens_d):
        ax2.annotate(f'{val:.2f}', xy=(val + 0.1, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_5_statistical_tests.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_5_statistical_tests.png")

# ============================================================
# PLOT 6: Confusion Matrix Heatmap (Section 6.4)
# ============================================================
def plot_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Federated aggregate confusion matrix
    fed_cm = np.array([
        [int(fed['tn'].sum()), int(fed['fp'].sum())],
        [int(fed['fn'].sum()), int(fed['tp'].sum())]
    ])

    sns.heatmap(fed_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
                yticklabels=['Actual Normal', 'Actual Pneumonia'])
    axes[0].set_title('Federated Learning (Aggregate, n=5 runs)')

    # Federated per-run heatmap
    run_data = fed[['run_id', 'tp', 'tn', 'fp', 'fn', 'recall']].copy()
    run_data['sensitivity'] = run_data['tp'] / (run_data['tp'] + run_data['fn'])
    run_data['specificity'] = run_data['tn'] / (run_data['tn'] + run_data['fp'])
    run_data['fnr'] = run_data['fn'] / (run_data['fn'] + run_data['tp'])
    run_data['fpr'] = run_data['fp'] / (run_data['fp'] + run_data['tn'])

    heatmap_data = run_data[['sensitivity', 'specificity', 'fnr', 'fpr']].values
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1],
                xticklabels=['Sensitivity', 'Specificity', 'FNR', 'FPR'],
                yticklabels=[f"Run {int(r)}" for r in run_data['run_id']])
    axes[1].set_title('Fairness Metrics by Run')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_6_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_6_confusion_matrices.png")

# ============================================================
# PLOT 7: FNR vs FPR Trade-off (Section 6.4)
# ============================================================
def plot_fnr_fpr_tradeoff():
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate per-run metrics
    fed_runs = []
    for _, row in fed.iterrows():
        tp, tn, fp, fn = row['tp'], row['tn'], row['fp'], row['fn']
        fed_runs.append({
            'run_id': int(row['run_id']),
            'fnr': fn / (fn + tp),
            'fpr': fp / (fp + tn)
        })

    # Scatter plot
    fnrs = [r['fnr'] for r in fed_runs]
    fprs = [r['fpr'] for r in fed_runs]

    ax.scatter(fprs, fnrs, s=200, c=FED_COLOR, alpha=0.7, edgecolors='black', linewidth=2, label='Federated Runs')

    # Add run labels
    for r in fed_runs:
        ax.annotate(f"Run {r['run_id']}", (r['fpr'] + 0.02, r['fnr'] + 0.005), fontsize=10)

    # Add centralized point (using recall as proxy)
    cent_fnr = 1 - cent['recall'].mean()
    ax.axhline(y=cent_fnr, color=CENT_COLOR, linestyle='--', linewidth=2, alpha=0.7, label=f'Centralized FNR ({cent_fnr:.3f})')

    # Risk zones
    ax.axhline(y=0.15, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=0.20, color='red', linestyle=':', alpha=0.5)
    ax.fill_between([0, 0.20], [0, 0], [0.15, 0.15], alpha=0.1, color='green', label='Low Risk Zone')
    ax.fill_between([0.20, 1], [0.15, 0.15], [1, 1], alpha=0.1, color='red', label='High Risk Zone')

    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('False Negative Rate (FNR)')
    ax.set_title('Clinical Risk Trade-off: FNR vs FPR')
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.2)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_7_fnr_fpr_tradeoff.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_7_fnr_fpr_tradeoff.png")

# ============================================================
# PLOT 8: Radar Chart - Overall Comparison (Summary)
# ============================================================
def plot_radar_comparison():
    categories = ['Recall', 'AUROC', 'Accuracy', 'Precision', 'F1-Score']
    N = len(categories)

    cent_vals = [
        cent['recall'].mean(),
        cent['auroc'].mean(),
        cent['accuracy'].mean(),
        cent['precision'].mean(),
        cent['f1_score'].mean()
    ]
    fed_vals = [
        fed['recall'].mean(),
        fed['auroc'].mean(),
        fed['accuracy'].mean(),
        fed['precision'].mean(),
        fed['f1_score'].mean()
    ]

    # Close the loop
    cent_vals += cent_vals[:1]
    fed_vals += fed_vals[:1]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, cent_vals, 'o-', linewidth=2, color=CENT_COLOR, label='Centralized')
    ax.fill(angles, cent_vals, alpha=0.25, color=CENT_COLOR)
    ax.plot(angles, fed_vals, 'o-', linewidth=2, color=FED_COLOR, label='Federated')
    ax.fill(angles, fed_vals, alpha=0.25, color=FED_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Comparison', size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_8_radar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_8_radar_comparison.png")

# ============================================================
# Run all plots
# ============================================================
if __name__ == "__main__":
    plot_metric_comparison()
    plot_recall_boxplot()
    plot_baseline_comparison()
    plot_cv_comparison()
    plot_statistical_tests()
    plot_confusion_matrices()
    plot_fnr_fpr_tradeoff()
    plot_radar_comparison()

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("\nGenerated files:")
    for f in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  - {f.name}")
