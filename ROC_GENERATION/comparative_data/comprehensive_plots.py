"""
Comprehensive Visualization Script for FYP2
=============================================
Generates all plots for Chapter 6: Results & Analysis

Plots Generated:
1. Fig 6.1: Metric Comparison Bar Chart (Section 6.2.1)
2. Fig 6.2: Recall Box Plot with Significance (Section 6.2.1)
3. Fig 6.3: Baseline Comparison (Section 6.2.2)
4. Fig 6.4: Coefficient of Variation (Section 6.2.3)
5. Fig 6.5: Statistical Significance - p-values & Cohen's d (Section 6.2.3)
6. Fig 6.6: Side-by-Side Confusion Matrix Heatmaps (Section 6.4)
7. Fig 6.7: FNR vs FPR Trade-off with Clinical Zones (Section 6.4)
8. Fig 6.8: Radar Chart - Overall Comparison
9. Fig 6.9: Per-Run Fairness Heatmap (Section 6.4)
10. Fig 6.10: Paired Difference Plot (Section 6.2.3)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150

# Colors
CENT_COLOR = '#2ecc71'    # Green
FED_COLOR = '#3498db'     # Blue
BASELINE_COLOR = '#e74c3c'  # Red
HIGHLIGHT = '#f39c12'     # Orange

SEEDS = [44, 45, 46, 47, 48]


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load analysis results and dataframes."""
    with open(BASE_DIR / "statistical_analysis.json") as f:
        stats = json.load(f)

    cent_df = pd.read_csv(BASE_DIR / "centralized" / "experiment_results.csv")
    cent_df = cent_df[cent_df['seed'].isin(SEEDS)].sort_values('seed').reset_index(drop=True)

    fed_df = pd.read_csv(BASE_DIR / "federated" / "federated_metrics_by_seed.csv")
    fed_df = fed_df[fed_df['seed'].isin(SEEDS)].sort_values('seed').reset_index(drop=True)

    with open(BASE_DIR / "baseline" / "pre_tuning_federated.json") as f:
        baseline_raw = json.load(f)
    baseline = eval(baseline_raw['evaluate_metrics_serverapp']['2'])

    return stats, cent_df, fed_df, baseline


# =============================================================================
# PLOT 1: Metric Comparison Bar Chart (Section 6.2.1)
# =============================================================================
def plot_metric_comparison(stats):
    """Grouped bar chart comparing all metrics with error bars."""
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
                   label='Centralized (n=5)', color=CENT_COLOR, capsize=5, alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width/2, fed_means, width, yerr=fed_stds,
                   label='Federated (n=5)', color=FED_COLOR, capsize=5, alpha=0.85, edgecolor='black')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics: Centralized vs Federated Learning\n(Seeds 44-48, 5 Paired Runs)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.3, label='Clinical threshold')

    # Add value labels
    for bar, val in zip(bars1, cent_means):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, fed_means):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    # Add significance markers
    for i, metric in enumerate(metrics):
        if stats['statistical_tests'][metric]['paired_t_test']['significant']:
            max_h = max(cent_means[i] + cent_stds[i], fed_means[i] + fed_stds[i])
            ax.annotate('*', xy=(i, max_h + 0.03), ha='center', fontsize=16, color='red')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_1_metric_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_1_metric_comparison.png")


# =============================================================================
# PLOT 2: Recall Box Plot with Significance (Section 6.2.1)
# =============================================================================
def plot_recall_boxplot(stats, cent_df, fed_df):
    """Box plot showing recall distribution with individual points."""
    fig, ax = plt.subplots(figsize=(8, 7))

    data = [cent_df['recall'].values, fed_df['recall'].values]
    positions = [1, 2]

    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(CENT_COLOR)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(FED_COLOR)
    bp['boxes'][1].set_alpha(0.7)

    # Add individual points with seed labels
    for i, (d, seeds) in enumerate(zip(data, [SEEDS, SEEDS])):
        x = np.random.normal(positions[i], 0.06, size=len(d))
        scatter = ax.scatter(x, d, alpha=0.8, color='black', s=80, zorder=3, edgecolors='white')
        for xi, yi, seed in zip(x, d, seeds):
            ax.annotate(f'{seed}', (xi + 0.08, yi), fontsize=8, alpha=0.7)

    ax.set_xticklabels(['Centralized\n(n=5)', 'Federated\n(n=5)'], fontsize=12)
    ax.set_ylabel('Recall (Sensitivity)', fontsize=12)
    ax.set_title('Recall Distribution by Training Paradigm', fontsize=14)
    ax.set_ylim(0.75, 1.02)

    # Add mean values
    cent_mean = cent_df['recall'].mean()
    fed_mean = fed_df['recall'].mean()
    ax.axhline(y=cent_mean, xmin=0.1, xmax=0.4, color=CENT_COLOR, linestyle='--', linewidth=2, alpha=0.8)
    ax.axhline(y=fed_mean, xmin=0.6, xmax=0.9, color=FED_COLOR, linestyle='--', linewidth=2, alpha=0.8)

    # Add significance annotation
    p_val = stats['statistical_tests']['recall']['paired_t_test']['p_value']
    sig_text = f'p = {p_val:.4f}' + (' *' if p_val < 0.05 else '')
    cohens_d = stats['statistical_tests']['recall']['cohens_d']

    # Draw bracket
    ax.plot([1, 1, 2, 2], [0.99, 1.0, 1.0, 0.99], 'k-', linewidth=1.5)
    ax.annotate(f'{sig_text}\nCohen\'s d = {cohens_d:.2f}', xy=(1.5, 1.005), ha='center', fontsize=10)

    # Add mean annotations
    ax.annotate(f'μ = {cent_mean:.3f}', xy=(0.7, cent_mean), fontsize=10, color=CENT_COLOR, fontweight='bold')
    ax.annotate(f'μ = {fed_mean:.3f}', xy=(2.15, fed_mean), fontsize=10, color=FED_COLOR, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_2_recall_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_2_recall_boxplot.png")


# =============================================================================
# PLOT 3: Baseline Comparison (Section 6.2.2)
# =============================================================================
def plot_baseline_comparison(stats, baseline):
    """Compare baseline, centralized tuned, and federated tuned."""
    metrics = ['recall', 'auroc', 'f1_score', 'accuracy']
    labels = ['Recall', 'AUROC', 'F1-Score', 'Accuracy']

    bc = stats['baseline_comparison']
    baseline_vals = [bc['baseline'][m] for m in metrics]
    cent_vals = [bc['centralized_tuned'][m] for m in metrics]
    fed_vals = [bc['federated_tuned'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width, baseline_vals, width, label='Baseline (Dec 26)',
                   color=BASELINE_COLOR, alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x, cent_vals, width, label='Tuned Centralized',
                   color=CENT_COLOR, alpha=0.85, edgecolor='black')
    bars3 = ax.bar(x + width, fed_vals, width, label='Tuned Federated',
                   color=FED_COLOR, alpha=0.85, edgecolor='black')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance: Baseline vs Tuned Models', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.2)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    # Add improvement arrows for recall
    recall_idx = 0
    base_recall = baseline_vals[recall_idx]
    fed_recall = fed_vals[recall_idx]
    improvement = bc['improvement_from_baseline']['recall']['federated_pct']

    ax.annotate('', xy=(x[recall_idx] + width, fed_recall),
               xytext=(x[recall_idx] - width, base_recall),
               arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.annotate(f'+{improvement:.1f}%', xy=(x[recall_idx], (base_recall + fed_recall)/2 + 0.05),
               fontsize=11, color='green', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_3_baseline_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_3_baseline_comparison.png")


# =============================================================================
# PLOT 4: Coefficient of Variation (Section 6.2.3)
# =============================================================================
def plot_cv_comparison(stats):
    """Bar chart showing CV% for model stability analysis."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']

    cv = stats['coefficient_of_variation']
    cent_cv = [cv['centralized'][m] for m in metrics]
    fed_cv = [cv['federated'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, cent_cv, width, label='Centralized',
                   color=CENT_COLOR, alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width/2, fed_cv, width, label='Federated',
                   color=FED_COLOR, alpha=0.85, edgecolor='black')

    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax.set_title('Model Stability: Coefficient of Variation by Metric\n(Lower = More Stable)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)

    # Add threshold lines
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.annotate('High variability (10%)', xy=(4.5, 10.5), fontsize=9, color='red')
    ax.annotate('Moderate (5%)', xy=(4.5, 5.5), fontsize=9, color='orange')

    # Add value labels
    for bar, val in zip(bars1, cent_cv):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, fed_cv):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_4_cv_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_4_cv_comparison.png")


# =============================================================================
# PLOT 5: Statistical Significance (Section 6.2.3)
# =============================================================================
def plot_statistical_tests(stats):
    """Dual panel: p-values and Cohen's d effect sizes."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']

    p_values = [stats['statistical_tests'][m]['paired_t_test']['p_value'] for m in metrics]
    cohens_d = [abs(stats['statistical_tests'][m]['cohens_d']) for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: P-values
    colors = ['#27ae60' if p < 0.05 else '#e74c3c' for p in p_values]
    bars1 = ax1.barh(labels, p_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax1.set_xlabel('p-value', fontsize=12)
    ax1.set_title('Statistical Significance (Paired t-test)', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, max(p_values) * 1.3)

    for bar, val in zip(bars1, p_values):
        sig = ' *' if val < 0.05 else ''
        ax1.annotate(f'{val:.4f}{sig}', xy=(val + 0.002, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10, fontweight='bold')

    # Panel 2: Cohen's d
    effect_colors = []
    for d in cohens_d:
        if d >= 0.8:
            effect_colors.append('#27ae60')  # Large - green
        elif d >= 0.5:
            effect_colors.append('#f39c12')  # Medium - orange
        else:
            effect_colors.append('#e74c3c')  # Small - red

    bars2 = ax2.barh(labels, cohens_d, color=effect_colors, alpha=0.8, edgecolor='black')
    ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("|Cohen's d|", fontsize=12)
    ax2.set_title("Effect Size (Cohen's d)", fontsize=13)

    # Legend for effect sizes
    legend_elements = [
        Patch(facecolor='#27ae60', label='Large (≥0.8)'),
        Patch(facecolor='#f39c12', label='Medium (0.5-0.8)'),
        Patch(facecolor='#e74c3c', label='Small (<0.5)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

    for bar, val, effect in zip(bars2, cohens_d, [stats['statistical_tests'][m]['effect_size'] for m in metrics]):
        ax2.annotate(f'{val:.2f} ({effect})', xy=(val + 0.05, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_5_statistical_tests.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_5_statistical_tests.png")


# =============================================================================
# PLOT 6: Confusion Matrix Heatmaps (Section 6.4)
# =============================================================================
def plot_confusion_matrices(stats):
    """Side-by-side normalized confusion matrix heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, paradigm in enumerate(['centralized', 'federated']):
        agg = stats['fairness_metrics'][paradigm]['aggregate']
        tp, tn, fp, fn = agg['tp'], agg['tn'], agg['fp'], agg['fn']

        # Create confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()

        # Normalize
        cm_norm = cm / total * 100

        # Plot
        sns.heatmap(cm_norm, annot=False, cmap='Blues' if paradigm == 'federated' else 'Greens',
                    ax=axes[idx], cbar=True, vmin=0, vmax=60,
                    xticklabels=['Predicted\nNormal', 'Predicted\nPneumonia'],
                    yticklabels=['Actual\nNormal', 'Actual\nPneumonia'])

        # Add custom annotations with counts and percentages
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_norm[i, j]
                axes[idx].annotate(f'{count:,.0f}\n({pct:.1f}%)',
                                  xy=(j + 0.5, i + 0.5), ha='center', va='center',
                                  fontsize=12, fontweight='bold', color='white' if pct > 30 else 'black')

        title_extra = '\n(Approximated)' if paradigm == 'centralized' else '\n(Actual)'
        axes[idx].set_title(f'{paradigm.title()} Confusion Matrix{title_extra}', fontsize=13)

    plt.suptitle('Aggregate Confusion Matrices (Seeds 44-48, 5 Runs)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_6_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_6_confusion_matrices.png")


# =============================================================================
# PLOT 7: FNR vs FPR Trade-off (Section 6.4)
# =============================================================================
def plot_fnr_fpr_tradeoff(stats):
    """Scatter plot showing FNR vs FPR with clinical risk zones."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot per-run federated points
    fed_runs = stats['fairness_metrics']['federated']['per_run']
    fnrs = [r['fnr'] for r in fed_runs]
    fprs = [r['fpr'] for r in fed_runs]
    seeds = [r['seed'] for r in fed_runs]

    ax.scatter(fprs, fnrs, s=200, c=FED_COLOR, alpha=0.8, edgecolors='black',
              linewidth=2, label='Federated Runs', zorder=5)

    for fpr, fnr, seed in zip(fprs, fnrs, seeds):
        ax.annotate(f'Seed {seed}', (fpr + 0.015, fnr + 0.003), fontsize=9)

    # Plot centralized aggregate point
    cent_agg = stats['fairness_metrics']['centralized']['aggregate']
    ax.scatter([cent_agg['fpr']], [cent_agg['fnr']], s=250, c=CENT_COLOR, marker='s',
              alpha=0.8, edgecolors='black', linewidth=2, label='Centralized (Approx)', zorder=5)
    ax.annotate('Centralized', (cent_agg['fpr'] + 0.015, cent_agg['fnr'] + 0.003), fontsize=9)

    # Risk zones
    ax.axhline(y=0.15, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.axhline(y=0.05, color='orange', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.axvline(x=0.20, color='red', linestyle=':', alpha=0.6, linewidth=1.5)

    # Fill zones
    ax.fill_between([0, 0.20], [0, 0], [0.05, 0.05], alpha=0.15, color='green', label='Low Risk Zone')
    ax.fill_between([0, 0.20], [0.05, 0.05], [0.15, 0.15], alpha=0.15, color='yellow', label='Moderate Risk')
    ax.fill_between([0, 1], [0.15, 0.15], [1, 1], alpha=0.15, color='red', label='High Risk Zone')

    # Labels
    ax.set_xlabel('False Positive Rate (FPR) - Unnecessary Treatment', fontsize=12)
    ax.set_ylabel('False Negative Rate (FNR) - Missed Pneumonia', fontsize=12)
    ax.set_title('Clinical Risk Trade-off: FNR vs FPR\n(Lower is Better for Both)', fontsize=14)
    ax.set_xlim(-0.02, 0.7)
    ax.set_ylim(-0.01, 0.20)
    ax.legend(loc='upper right', fontsize=10)

    # Add clinical interpretation
    ax.annotate('← Clinical Priority:\nMinimize FNR', xy=(0.02, 0.18), fontsize=10,
               color='darkred', style='italic')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_7_fnr_fpr_tradeoff.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_7_fnr_fpr_tradeoff.png")


# =============================================================================
# PLOT 8: Radar Chart (Summary)
# =============================================================================
def plot_radar_comparison(stats):
    """Radar/spider chart for overall profile comparison."""
    categories = ['Recall', 'AUROC', 'Accuracy', 'Precision', 'F1-Score']
    metrics = ['recall', 'auroc', 'accuracy', 'precision', 'f1_score']
    N = len(categories)

    cent_vals = [stats['summary_stats']['centralized'][m]['mean'] for m in metrics]
    fed_vals = [stats['summary_stats']['federated'][m]['mean'] for m in metrics]

    # Close the loop
    cent_vals += cent_vals[:1]
    fed_vals += fed_vals[:1]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    ax.plot(angles, cent_vals, 'o-', linewidth=2.5, color=CENT_COLOR, label='Centralized', markersize=8)
    ax.fill(angles, cent_vals, alpha=0.25, color=CENT_COLOR)
    ax.plot(angles, fed_vals, 'o-', linewidth=2.5, color=FED_COLOR, label='Federated', markersize=8)
    ax.fill(angles, fed_vals, alpha=0.25, color=FED_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Profile Comparison\n(Seeds 44-48)', size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=11)

    # Add value annotations
    for angle, cent_val, fed_val in zip(angles[:-1], cent_vals[:-1], fed_vals[:-1]):
        ax.annotate(f'{cent_val:.3f}', xy=(angle, cent_val + 0.05), fontsize=8, color=CENT_COLOR, ha='center')
        ax.annotate(f'{fed_val:.3f}', xy=(angle, fed_val - 0.08), fontsize=8, color=FED_COLOR, ha='center')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_8_radar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_8_radar_comparison.png")


# =============================================================================
# PLOT 9: Per-Run Fairness Heatmap (Section 6.4)
# =============================================================================
def plot_fairness_heatmap(stats):
    """Heatmap showing fairness metrics per run for federated."""
    fed_runs = stats['fairness_metrics']['federated']['per_run']

    data = []
    for run in fed_runs:
        data.append([run['sensitivity'], run['specificity'], run['fnr'], run['fpr']])

    df = pd.DataFrame(data, columns=['Sensitivity', 'Specificity', 'FNR', 'FPR'],
                     index=[f"Seed {r['seed']}" for r in fed_runs])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Custom colormap: green for good, red for bad
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, center=0.5,
                linewidths=0.5, cbar_kws={'label': 'Score'}, vmin=0, vmax=1)

    ax.set_title('Per-Run Fairness Metrics (Federated)\nGreen = Better, Red = Worse', fontsize=13)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Run', fontsize=12)

    # Highlight FNR column (critical metric)
    for i in range(len(fed_runs)):
        ax.add_patch(plt.Rectangle((2, i), 1, 1, fill=False, edgecolor='black', linewidth=2))

    ax.annotate('← Critical Metric', xy=(3.1, 2.5), fontsize=10, color='darkred')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_9_fairness_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_9_fairness_heatmap.png")


# =============================================================================
# PLOT 10: Paired Difference Plot (Section 6.2.3)
# =============================================================================
def plot_paired_differences(stats, cent_df, fed_df):
    """Paired difference plot showing per-seed comparison."""
    metrics = ['recall', 'accuracy', 'precision', 'f1_score', 'auroc']
    labels = ['Recall', 'Accuracy', 'Precision', 'F1-Score', 'AUROC']

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]

        cent_vals = cent_df[metric].values
        fed_vals = fed_df[metric].values
        diff = cent_vals - fed_vals

        # Plot paired lines
        for i, seed in enumerate(SEEDS):
            ax.plot([0, 1], [cent_vals[i], fed_vals[i]], 'o-', color='gray', alpha=0.5, linewidth=1)
            ax.scatter([0], [cent_vals[i]], color=CENT_COLOR, s=80, zorder=5)
            ax.scatter([1], [fed_vals[i]], color=FED_COLOR, s=80, zorder=5)
            ax.annotate(f'{seed}', (0.02, cent_vals[i]), fontsize=8, alpha=0.7)

        # Add mean lines
        ax.axhline(y=cent_vals.mean(), xmin=0.05, xmax=0.35, color=CENT_COLOR, linestyle='--', linewidth=2)
        ax.axhline(y=fed_vals.mean(), xmin=0.65, xmax=0.95, color=FED_COLOR, linestyle='--', linewidth=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Centralized', 'Federated'])
        ax.set_ylabel(label)
        ax.set_title(f'{label}\n(Δ = {diff.mean():+.4f})')

        # Add significance marker
        p_val = stats['statistical_tests'][metric]['paired_t_test']['p_value']
        if p_val < 0.05:
            ax.annotate('*', xy=(0.5, ax.get_ylim()[1]), fontsize=16, color='red', ha='center')

    # Remove extra subplot
    axes[5].axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CENT_COLOR, markersize=10, label='Centralized'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=FED_COLOR, markersize=10, label='Federated'),
        plt.Line2D([0], [0], color='gray', alpha=0.5, label='Paired connection')
    ]
    axes[5].legend(handles=legend_elements, loc='center', fontsize=12)
    axes[5].set_title('Legend', fontsize=12)

    plt.suptitle('Paired Comparison by Seed (Lines connect same seed)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_10_paired_differences.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_10_paired_differences.png")


# =============================================================================
# MAIN
# =============================================================================
def generate_all_plots():
    """Generate all plots for Chapter 6."""
    print("=" * 60)
    print("Generating Comprehensive Plots for Chapter 6")
    print("=" * 60)

    print("\nLoading data...")
    stats, cent_df, fed_df, baseline = load_data()

    print("\nGenerating plots...")
    plot_metric_comparison(stats)
    plot_recall_boxplot(stats, cent_df, fed_df)
    plot_baseline_comparison(stats, baseline)
    plot_cv_comparison(stats)
    plot_statistical_tests(stats)
    plot_confusion_matrices(stats)
    plot_fnr_fpr_tradeoff(stats)
    plot_radar_comparison(stats)
    plot_fairness_heatmap(stats)
    plot_paired_differences(stats, cent_df, fed_df)

    print(f"\n{'=' * 60}")
    print(f"All plots saved to: {PLOTS_DIR}")
    print(f"{'=' * 60}")

    print("\nGenerated files:")
    for f in sorted(PLOTS_DIR.glob("fig_*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    generate_all_plots()
