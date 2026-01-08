"""
Comprehensive Visualization Script for FYP2 - CentralizedV1 Version
=====================================================================
Generates all plots for Chapter 6: Results & Analysis
Uses CentralizedV1 data (seeds 42-46) paired with Federated (seeds 44-48)

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
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent
CENTRALIZEDV1_DIR = Path(__file__).parent.parent / "results" / "CentralizedV1"
PLOTS_DIR = BASE_DIR / "plots_v1"
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

# Seeds mapping: centralized 42-46 paired with federated 44-48
CENT_SEEDS = [42, 43, 44, 45, 46]
FED_SEEDS = [44, 45, 46, 47, 48]
PAIRED_LABELS = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']


# =============================================================================
# DATA LOADING & STATISTICS REGENERATION
# =============================================================================
def load_data():
    """Load data from CentralizedV1 and federated directories."""
    # Load CentralizedV1 data
    cent_df = pd.read_csv(CENTRALIZEDV1_DIR / "experiment_results.csv")
    cent_df = cent_df[cent_df['seed'].isin(CENT_SEEDS)].sort_values('seed').reset_index(drop=True)

    # Load confusion matrices
    with open(CENTRALIZEDV1_DIR / "confusion_matrices.json") as f:
        cent_cm = json.load(f)

    # Load federated data
    fed_df = pd.read_csv(BASE_DIR / "federated" / "federated_metrics_by_seed.csv")
    fed_df = fed_df[fed_df['seed'].isin(FED_SEEDS)].sort_values('seed').reset_index(drop=True)

    # Load baseline
    with open(BASE_DIR / "baseline" / "pre_tuning_federated.json") as f:
        baseline_raw = json.load(f)
    baseline = eval(baseline_raw['evaluate_metrics_serverapp']['2'])

    return cent_df, fed_df, cent_cm, baseline


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def get_effect_size_label(d):
    """Get effect size label from Cohen's d."""
    d = abs(d)
    if d >= 0.8:
        return "large"
    elif d >= 0.5:
        return "medium"
    else:
        return "small"


def regenerate_statistics(cent_df, fed_df, cent_cm):
    """Regenerate all statistics from raw data."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']

    stats = {
        'metadata': {
            'centralized_seeds': CENT_SEEDS,
            'federated_seeds': FED_SEEDS,
            'n_runs_per_paradigm': 5,
            'pairing_note': 'Seeds paired by run order (42↔44, 43↔45, etc.)'
        },
        'summary_stats': {'centralized': {}, 'federated': {}},
        'statistical_tests': {},
        'coefficient_of_variation': {'centralized': {}, 'federated': {}},
        'fairness_metrics': {'centralized': {}, 'federated': {}},
        'baseline_comparison': {}
    }

    # Summary stats
    for paradigm, df in [('centralized', cent_df), ('federated', fed_df)]:
        for metric in metrics:
            vals = df[metric].values
            stats['summary_stats'][paradigm][metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'n': len(vals),
                'values': vals.tolist()
            }

    # Statistical tests (paired by run order)
    for metric in metrics:
        cent_vals = cent_df[metric].values
        fed_vals = fed_df[metric].values

        # Paired t-test
        t_stat, p_val = scipy_stats.ttest_rel(cent_vals, fed_vals)

        # Cohen's d
        d = calculate_cohens_d(cent_vals, fed_vals)

        stats['statistical_tests'][metric] = {
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05)
            },
            'cohens_d': float(d),
            'effect_size': get_effect_size_label(d)
        }

    # Coefficient of variation
    for paradigm, df in [('centralized', cent_df), ('federated', fed_df)]:
        for metric in metrics:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            cv = (std_val / mean_val * 100) if mean_val > 0 else 0
            stats['coefficient_of_variation'][paradigm][metric] = round(cv, 2)

    # Fairness metrics from confusion matrices
    cent_runs = cent_cm['results']
    cent_tp = sum(r['tp'] for r in cent_runs)
    cent_tn = sum(r['tn'] for r in cent_runs)
    cent_fp = sum(r['fp'] for r in cent_runs)
    cent_fn = sum(r['fn'] for r in cent_runs)

    stats['fairness_metrics']['centralized'] = {
        'aggregate': {
            'tp': cent_tp, 'tn': cent_tn, 'fp': cent_fp, 'fn': cent_fn,
            'sensitivity': cent_tp / (cent_tp + cent_fn) if (cent_tp + cent_fn) > 0 else 0,
            'specificity': cent_tn / (cent_tn + cent_fp) if (cent_tn + cent_fp) > 0 else 0,
            'fnr': cent_fn / (cent_fn + cent_tp) if (cent_fn + cent_tp) > 0 else 0,
            'fpr': cent_fp / (cent_fp + cent_tn) if (cent_fp + cent_tn) > 0 else 0
        },
        'per_run': [{
            'seed': r['seed'],
            'tp': r['tp'], 'tn': r['tn'], 'fp': r['fp'], 'fn': r['fn'],
            'sensitivity': r['recall'],
            'specificity': r['specificity'],
            'fnr': 1 - r['recall'],
            'fpr': 1 - r['specificity']
        } for r in cent_runs]
    }

    # Federated fairness from CSV
    fed_tp = int(fed_df['tp'].sum())
    fed_tn = int(fed_df['tn'].sum())
    fed_fp = int(fed_df['fp'].sum())
    fed_fn = int(fed_df['fn'].sum())

    stats['fairness_metrics']['federated'] = {
        'aggregate': {
            'tp': fed_tp, 'tn': fed_tn, 'fp': fed_fp, 'fn': fed_fn,
            'sensitivity': fed_tp / (fed_tp + fed_fn) if (fed_tp + fed_fn) > 0 else 0,
            'specificity': fed_tn / (fed_tn + fed_fp) if (fed_tn + fed_fp) > 0 else 0,
            'fnr': fed_fn / (fed_fn + fed_tp) if (fed_fn + fed_tp) > 0 else 0,
            'fpr': fed_fp / (fed_fp + fed_tn) if (fed_fp + fed_tn) > 0 else 0
        },
        'per_run': [{
            'seed': int(row['seed']),
            'tp': int(row['tp']), 'tn': int(row['tn']),
            'fp': int(row['fp']), 'fn': int(row['fn']),
            'sensitivity': row['recall'],
            'specificity': row['tn'] / (row['tn'] + row['fp']) if (row['tn'] + row['fp']) > 0 else 0,
            'fnr': 1 - row['recall'],
            'fpr': row['fp'] / (row['fp'] + row['tn']) if (row['fp'] + row['tn']) > 0 else 0
        } for _, row in fed_df.iterrows()]
    }

    return stats


def add_baseline_comparison(stats, baseline):
    """Add baseline comparison to stats."""
    stats['baseline_comparison'] = {
        'baseline': {
            'recall': baseline['server_recall'],
            'auroc': baseline['server_auroc'],
            'f1_score': baseline['server_f1'],
            'accuracy': baseline['server_accuracy']
        },
        'centralized_tuned': {
            'recall': stats['summary_stats']['centralized']['recall']['mean'],
            'auroc': stats['summary_stats']['centralized']['auroc']['mean'],
            'f1_score': stats['summary_stats']['centralized']['f1_score']['mean'],
            'accuracy': stats['summary_stats']['centralized']['accuracy']['mean']
        },
        'federated_tuned': {
            'recall': stats['summary_stats']['federated']['recall']['mean'],
            'auroc': stats['summary_stats']['federated']['auroc']['mean'],
            'f1_score': stats['summary_stats']['federated']['f1_score']['mean'],
            'accuracy': stats['summary_stats']['federated']['accuracy']['mean']
        }
    }

    # Calculate improvements
    base = stats['baseline_comparison']['baseline']
    cent = stats['baseline_comparison']['centralized_tuned']
    fed = stats['baseline_comparison']['federated_tuned']

    stats['baseline_comparison']['improvement_from_baseline'] = {}
    for metric in ['recall', 'auroc', 'f1_score', 'accuracy']:
        base_val = base[metric]
        cent_pct = ((cent[metric] - base_val) / base_val * 100) if base_val > 0 else 0
        fed_pct = ((fed[metric] - base_val) / base_val * 100) if base_val > 0 else 0
        stats['baseline_comparison']['improvement_from_baseline'][metric] = {
            'centralized_pct': round(cent_pct, 2),
            'federated_pct': round(fed_pct, 2)
        }

    return stats


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
    ax.set_title('Performance Metrics: Centralized vs Federated Learning\n(CentralizedV1: Seeds 42-46, Federated: Seeds 44-48)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.3, label='Clinical threshold')

    # Add value labels ABOVE error bars (use mean + std + offset)
    for i, (bar, val, std) in enumerate(zip(bars1, cent_means, cent_stds)):
        label_y = val + std + 0.02  # Position above error bar
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, label_y),
                   xytext=(0, 0), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    for i, (bar, val, std) in enumerate(zip(bars2, fed_means, fed_stds)):
        label_y = val + std + 0.02  # Position above error bar
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, label_y),
                   xytext=(0, 0), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    # Add significance markers (even higher)
    for i, metric in enumerate(metrics):
        if stats['statistical_tests'][metric]['paired_t_test']['significant']:
            max_h = max(cent_means[i] + cent_stds[i], fed_means[i] + fed_stds[i])
            ax.annotate('*', xy=(i, max_h + 0.06), ha='center', fontsize=16, color='red')

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

    # Add individual points with run labels
    for i, (d, labels) in enumerate(zip(data, [PAIRED_LABELS, PAIRED_LABELS])):
        x = np.random.normal(positions[i], 0.06, size=len(d))
        ax.scatter(x, d, alpha=0.8, color='black', s=80, zorder=3, edgecolors='white')
        for xi, yi, label in zip(x, d, labels):
            ax.annotate(label, (xi + 0.08, yi), fontsize=8, alpha=0.7)

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
def plot_baseline_comparison(stats):
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

    # Add improvement visualization for recall - more prominent
    recall_idx = 0
    base_recall = baseline_vals[recall_idx]
    cent_recall = cent_vals[recall_idx]
    fed_recall = fed_vals[recall_idx]
    fed_improvement = bc['improvement_from_baseline']['recall']['federated_pct']
    cent_improvement = bc['improvement_from_baseline']['recall']['centralized_pct']

    # Draw a prominent bracket/arc from baseline to federated
    # First, draw vertical lines from each bar
    bracket_y = max(base_recall, cent_recall, fed_recall) + 0.08

    # Horizontal bracket line
    ax.plot([x[recall_idx] - width, x[recall_idx] + width], [bracket_y, bracket_y],
           color='#27ae60', linewidth=3, solid_capstyle='round')

    # Vertical connectors
    ax.plot([x[recall_idx] - width, x[recall_idx] - width], [base_recall + 0.02, bracket_y],
           color='#27ae60', linewidth=2, linestyle='-')
    ax.plot([x[recall_idx] + width, x[recall_idx] + width], [fed_recall + 0.02, bracket_y],
           color='#27ae60', linewidth=2, linestyle='-')

    # Big arrow pointing up with improvement text
    ax.annotate(f'+{fed_improvement:.1f}%\nRecall Gain',
               xy=(x[recall_idx], bracket_y + 0.02),
               fontsize=12, color='#27ae60', fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2))

    # Add a shaded region to highlight recall column
    ax.axvspan(x[recall_idx] - 0.4, x[recall_idx] + 0.4, alpha=0.08, color='green', zorder=0)

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

        axes[idx].set_title(f'{paradigm.title()} Confusion Matrix\n(Aggregate)', fontsize=13)

    plt.suptitle('Aggregate Confusion Matrices (5 Runs Each)', fontsize=14, y=1.02)
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

    # Plot per-run centralized points
    cent_runs = stats['fairness_metrics']['centralized']['per_run']
    cent_fnrs = [r['fnr'] for r in cent_runs]
    cent_fprs = [r['fpr'] for r in cent_runs]
    cent_seeds = [r['seed'] for r in cent_runs]

    ax.scatter(cent_fprs, cent_fnrs, s=200, c=CENT_COLOR, marker='s', alpha=0.8,
              edgecolors='black', linewidth=2, label='Centralized Runs', zorder=5)

    for fpr, fnr, seed in zip(cent_fprs, cent_fnrs, cent_seeds):
        ax.annotate(f'Seed {seed}', (fpr + 0.015, fnr + 0.003), fontsize=9)

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
    ax.set_xlim(-0.02, 0.85)
    ax.set_ylim(-0.01, 0.25)
    ax.legend(loc='upper right', fontsize=10)

    # Add clinical interpretation
    ax.annotate('← Clinical Priority:\nMinimize FNR', xy=(0.02, 0.23), fontsize=10,
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
    ax.set_title('Overall Performance Profile Comparison\n(CentralizedV1 vs Federated)', size=14, y=1.08)
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
    """Heatmap showing fairness metrics per run for both paradigms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (paradigm, ax) in enumerate(zip(['centralized', 'federated'], axes)):
        runs = stats['fairness_metrics'][paradigm]['per_run']

        data = []
        for run in runs:
            data.append([run['sensitivity'], run['specificity'], run['fnr'], run['fpr']])

        df = pd.DataFrame(data, columns=['Sensitivity', 'Specificity', 'FNR', 'FPR'],
                         index=[f"Seed {r['seed']}" for r in runs])

        # Custom colormap: green for good, red for bad
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, center=0.5,
                    linewidths=0.5, cbar_kws={'label': 'Score'}, vmin=0, vmax=1)

        ax.set_title(f'Per-Run Fairness Metrics ({paradigm.title()})\nGreen = Better, Red = Worse', fontsize=13)
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Run', fontsize=12)

        # Highlight FNR column (critical metric)
        for i in range(len(runs)):
            ax.add_patch(plt.Rectangle((2, i), 1, 1, fill=False, edgecolor='black', linewidth=2))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_9_fairness_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_9_fairness_heatmap.png")


# =============================================================================
# PLOT 10: Paired Difference Plot (Section 6.2.3)
# =============================================================================
def plot_paired_differences(stats, cent_df, fed_df):
    """Paired difference plot showing per-run comparison."""
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
        for i, run_label in enumerate(PAIRED_LABELS):
            ax.plot([0, 1], [cent_vals[i], fed_vals[i]], 'o-', color='gray', alpha=0.5, linewidth=1)
            ax.scatter([0], [cent_vals[i]], color=CENT_COLOR, s=80, zorder=5)
            ax.scatter([1], [fed_vals[i]], color=FED_COLOR, s=80, zorder=5)
            ax.annotate(run_label, (0.02, cent_vals[i]), fontsize=8, alpha=0.7)

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

    plt.suptitle('Paired Comparison by Run Order\n(Lines connect paired runs: Cent 42↔Fed 44, etc.)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_6_10_paired_differences.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: fig_6_10_paired_differences.png")


# =============================================================================
# MAIN
# =============================================================================
def generate_all_plots():
    """Generate all plots for Chapter 6 using CentralizedV1 data."""
    print("=" * 60)
    print("Generating Comprehensive Plots for Chapter 6")
    print("Using CentralizedV1 Data")
    print("=" * 60)

    print("\nLoading data...")
    cent_df, fed_df, cent_cm, baseline = load_data()

    print(f"  Centralized: {len(cent_df)} runs (seeds {CENT_SEEDS})")
    print(f"  Federated: {len(fed_df)} runs (seeds {FED_SEEDS})")

    print("\nRegenerating statistics...")
    stats = regenerate_statistics(cent_df, fed_df, cent_cm)
    stats = add_baseline_comparison(stats, baseline)

    # Save regenerated stats
    with open(PLOTS_DIR / "statistical_analysis_v1.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print("  Saved: statistical_analysis_v1.json")

    print("\nGenerating plots...")
    plot_metric_comparison(stats)
    plot_recall_boxplot(stats, cent_df, fed_df)
    plot_baseline_comparison(stats)
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
