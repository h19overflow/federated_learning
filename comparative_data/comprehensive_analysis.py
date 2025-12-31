"""
Comprehensive Statistical Analysis for FYP2
============================================
Centralized vs Federated Learning Comparison (Seeds 44-48, 5 Paired Runs)

Implements:
- Option A: Paired comparison using metrics (accuracy, precision, recall, F1, AUROC)
- Option C: Approximate centralized confusion matrix from recall/precision + dataset size

Generates outputs for SECTIONS.md:
- Section 6.2.1: Evaluation Metrics
- Section 6.2.2: Baseline Comparison
- Section 6.2.3: Cross-Validation & Variance Analysis
- Section 6.4: Bias & Fairness Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent
SEEDS_TO_COMPARE = [44, 45, 46, 47, 48]  # 5 paired runs
VAL_SAMPLES = 6046  # From validation_results.json
VAL_CLASS_0 = 4135  # Normal (from validation split)
VAL_CLASS_1 = 1911  # Pneumonia (from validation split)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load centralized and federated results for seeds 44-48."""
    # Centralized data
    cent_df = pd.read_csv(BASE_DIR / "centralized" / "experiment_results.csv")
    cent_df = cent_df[cent_df['seed'].isin(SEEDS_TO_COMPARE)].copy()
    cent_df = cent_df.sort_values('seed').reset_index(drop=True)

    # Federated data
    fed_df = pd.read_csv(BASE_DIR / "federated" / "federated_metrics_by_seed.csv")
    fed_df = fed_df[fed_df['seed'].isin(SEEDS_TO_COMPARE)].copy()
    fed_df = fed_df.sort_values('seed').reset_index(drop=True)

    # Load baseline
    with open(BASE_DIR / "baseline" / "pre_tuning_federated.json") as f:
        baseline_raw = json.load(f)
    baseline = eval(baseline_raw['evaluate_metrics_serverapp']['2'])

    # Load best run epochs for centralized (has confusion matrix)
    best_run_epochs = pd.read_csv(BASE_DIR / "centralized" / "best_run_epochs.csv")

    return cent_df, fed_df, baseline, best_run_epochs


# =============================================================================
# OPTION C: APPROXIMATE CENTRALIZED CONFUSION MATRIX
# =============================================================================
def approximate_confusion_matrix(recall, precision, total_positive, total_negative):
    """
    Approximate confusion matrix from recall and precision.

    Given:
    - Recall = TP / (TP + FN) => TP = Recall * (TP + FN) = Recall * total_positive
    - Precision = TP / (TP + FP) => FP = TP * (1 - Precision) / Precision
    - FN = total_positive - TP
    - TN = total_negative - FP
    """
    tp = recall * total_positive
    fn = total_positive - tp

    if precision > 0:
        fp = tp * (1 - precision) / precision
    else:
        fp = total_negative  # Worst case

    tn = total_negative - fp

    # Ensure non-negative
    tp = max(0, tp)
    fn = max(0, fn)
    fp = max(0, fp)
    tn = max(0, tn)

    return int(round(tp)), int(round(tn)), int(round(fp)), int(round(fn))


def add_centralized_confusion_matrix(cent_df):
    """Add approximated confusion matrix to centralized dataframe."""
    cms = []
    for _, row in cent_df.iterrows():
        tp, tn, fp, fn = approximate_confusion_matrix(
            row['recall'], row['precision'], VAL_CLASS_1, VAL_CLASS_0
        )
        cms.append({'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})

    cm_df = pd.DataFrame(cms)
    cent_df['tp_approx'] = cm_df['tp']
    cent_df['tn_approx'] = cm_df['tn']
    cent_df['fp_approx'] = cm_df['fp']
    cent_df['fn_approx'] = cm_df['fn']

    return cent_df


# =============================================================================
# SECTION 6.2.1: DESCRIPTIVE STATISTICS
# =============================================================================
def compute_descriptive_stats(cent_df, fed_df):
    """Compute mean, std, min, max, n for all metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']

    results = {'centralized': {}, 'federated': {}}

    for metric in metrics:
        for name, df in [('centralized', cent_df), ('federated', fed_df)]:
            values = df[metric].values
            results[name][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n': len(values),
                'values': values.tolist()
            }

    return results


# =============================================================================
# SECTION 6.2.3: STATISTICAL TESTS
# =============================================================================
def compute_statistical_tests(cent_df, fed_df):
    """Compute paired t-test, Wilcoxon, Cohen's d, and 95% CI."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']
    results = {}

    for metric in metrics:
        cent_vals = cent_df[metric].values
        fed_vals = fed_df[metric].values

        # Paired t-test (same seeds)
        t_stat, p_value_ttest = stats.ttest_rel(cent_vals, fed_vals)

        # Welch's t-test (independent, unequal variance)
        t_stat_welch, p_value_welch = stats.ttest_ind(cent_vals, fed_vals, equal_var=False)

        # Wilcoxon signed-rank test (non-parametric, n=5 is small)
        try:
            w_stat, p_value_wilcoxon = stats.wilcoxon(cent_vals, fed_vals)
        except ValueError:
            w_stat, p_value_wilcoxon = np.nan, np.nan

        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.var(cent_vals, ddof=1) + np.var(fed_vals, ddof=1)) / 2)
        cohens_d = (np.mean(cent_vals) - np.mean(fed_vals)) / pooled_std if pooled_std > 0 else 0

        # Effect size interpretation
        d_abs = abs(cohens_d)
        if d_abs < 0.2:
            effect_interpretation = "negligible"
        elif d_abs < 0.5:
            effect_interpretation = "small"
        elif d_abs < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        # 95% Confidence Interval for mean difference
        diff = cent_vals - fed_vals
        mean_diff = np.mean(diff)
        se_diff = stats.sem(diff)
        ci_95 = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=se_diff)

        results[metric] = {
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value_ttest),
                'significant': bool(p_value_ttest < 0.05)
            },
            'welch_t_test': {
                't_statistic': float(t_stat_welch),
                'p_value': float(p_value_welch),
                'significant': bool(p_value_welch < 0.05)
            },
            'wilcoxon_test': {
                'w_statistic': float(w_stat) if not np.isnan(w_stat) else None,
                'p_value': float(p_value_wilcoxon) if not np.isnan(p_value_wilcoxon) else None,
                'significant': bool(p_value_wilcoxon < 0.05) if not np.isnan(p_value_wilcoxon) else None
            },
            'cohens_d': float(cohens_d),
            'effect_size': effect_interpretation,
            'mean_difference': {
                'value': float(mean_diff),
                'ci_lower': float(ci_95[0]),
                'ci_upper': float(ci_95[1]),
                'centralized_higher': bool(mean_diff > 0)
            }
        }

    return results


# =============================================================================
# SECTION 6.2.3: COEFFICIENT OF VARIATION
# =============================================================================
def compute_coefficient_of_variation(stats_dict):
    """Compute CV% = (std/mean) * 100 for stability analysis."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']
    cv_results = {'centralized': {}, 'federated': {}}

    for paradigm in ['centralized', 'federated']:
        for metric in metrics:
            mean = stats_dict[paradigm][metric]['mean']
            std = stats_dict[paradigm][metric]['std']
            cv = (std / mean * 100) if mean > 0 else 0
            cv_results[paradigm][metric] = round(cv, 2)

    return cv_results


# =============================================================================
# SECTION 6.4: FAIRNESS METRICS
# =============================================================================
def compute_fairness_metrics(cent_df, fed_df):
    """Compute sensitivity, specificity, FNR, FPR for both paradigms."""
    results = {}

    # Federated (has actual confusion matrix)
    fed_tp = fed_df['tp'].sum()
    fed_tn = fed_df['tn'].sum()
    fed_fp = fed_df['fp'].sum()
    fed_fn = fed_df['fn'].sum()

    results['federated'] = {
        'aggregate': {
            'tp': int(fed_tp),
            'tn': int(fed_tn),
            'fp': int(fed_fp),
            'fn': int(fed_fn),
            'sensitivity': float(fed_tp / (fed_tp + fed_fn)) if (fed_tp + fed_fn) > 0 else 0,
            'specificity': float(fed_tn / (fed_tn + fed_fp)) if (fed_tn + fed_fp) > 0 else 0,
            'fnr': float(fed_fn / (fed_fn + fed_tp)) if (fed_fn + fed_tp) > 0 else 0,
            'fpr': float(fed_fp / (fed_fp + fed_tn)) if (fed_fp + fed_tn) > 0 else 0
        },
        'per_run': []
    }

    # Per-run federated metrics
    for _, row in fed_df.iterrows():
        tp, tn, fp, fn = row['tp'], row['tn'], row['fp'], row['fn']
        results['federated']['per_run'].append({
            'seed': int(row['seed']),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        })

    # Centralized (approximated confusion matrix)
    cent_tp = cent_df['tp_approx'].sum()
    cent_tn = cent_df['tn_approx'].sum()
    cent_fp = cent_df['fp_approx'].sum()
    cent_fn = cent_df['fn_approx'].sum()

    results['centralized'] = {
        'aggregate': {
            'tp': int(cent_tp),
            'tn': int(cent_tn),
            'fp': int(cent_fp),
            'fn': int(cent_fn),
            'sensitivity': float(cent_tp / (cent_tp + cent_fn)) if (cent_tp + cent_fn) > 0 else 0,
            'specificity': float(cent_tn / (cent_tn + cent_fp)) if (cent_tn + cent_fp) > 0 else 0,
            'fnr': float(cent_fn / (cent_fn + cent_tp)) if (cent_fn + cent_tp) > 0 else 0,
            'fpr': float(cent_fp / (cent_fp + cent_tn)) if (cent_fp + cent_tn) > 0 else 0,
            'note': 'Approximated from recall/precision using validation set class distribution'
        },
        'per_run': []
    }

    # Per-run centralized metrics (approximated)
    for _, row in cent_df.iterrows():
        tp = row['tp_approx']
        tn = row['tn_approx']
        fp = row['fp_approx']
        fn = row['fn_approx']
        results['centralized']['per_run'].append({
            'seed': int(row['seed']),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            'approximated': True
        })

    return results


def assess_clinical_risk(fairness_metrics):
    """Categorize clinical risk based on FNR and FPR thresholds."""
    risk_assessment = {}

    for paradigm in ['centralized', 'federated']:
        fnr = fairness_metrics[paradigm]['aggregate']['fnr']
        fpr = fairness_metrics[paradigm]['aggregate']['fpr']

        # FNR risk
        if fnr < 0.05:
            fnr_risk = "Low"
        elif fnr < 0.15:
            fnr_risk = "Moderate"
        else:
            fnr_risk = "High"

        # FPR risk
        if fpr < 0.10:
            fpr_risk = "Low"
        elif fpr < 0.20:
            fpr_risk = "Moderate"
        else:
            fpr_risk = "High"

        # Overall clinical assessment
        if fnr_risk == "Low" and fpr_risk in ["Low", "Moderate"]:
            overall = "Acceptable for clinical screening"
        elif fnr_risk == "High":
            overall = "Unacceptable - High missed diagnosis rate"
        else:
            overall = "Requires further validation"

        risk_assessment[paradigm] = {
            'fnr_risk': fnr_risk,
            'fpr_risk': fpr_risk,
            'overall_assessment': overall,
            'fnr_value': fnr,
            'fpr_value': fpr
        }

    return risk_assessment


# =============================================================================
# SECTION 6.2.2: BASELINE COMPARISON
# =============================================================================
def compute_baseline_comparison(cent_df, fed_df, baseline):
    """Compare tuned models against pre-tuning baseline."""
    metrics = ['recall', 'auroc', 'f1_score', 'accuracy']
    baseline_keys = ['server_recall', 'server_auroc', 'server_f1', 'server_accuracy']

    results = {
        'baseline': {},
        'centralized_tuned': {},
        'federated_tuned': {},
        'improvement_from_baseline': {}
    }

    for metric, baseline_key in zip(metrics, baseline_keys):
        baseline_val = baseline[baseline_key]
        cent_mean = cent_df[metric if metric != 'f1_score' else 'f1_score'].mean()
        fed_mean = fed_df[metric if metric != 'f1_score' else 'f1_score'].mean()

        results['baseline'][metric] = float(baseline_val)
        results['centralized_tuned'][metric] = float(cent_mean)
        results['federated_tuned'][metric] = float(fed_mean)

        # Improvement calculation
        cent_improvement = ((cent_mean - baseline_val) / baseline_val) * 100 if baseline_val > 0 else 0
        fed_improvement = ((fed_mean - baseline_val) / baseline_val) * 100 if baseline_val > 0 else 0

        results['improvement_from_baseline'][metric] = {
            'centralized_pct': round(cent_improvement, 2),
            'federated_pct': round(fed_improvement, 2),
            'centralized_better': cent_improvement > fed_improvement
        }

    return results


# =============================================================================
# GENERATE SUMMARY TABLES (MARKDOWN)
# =============================================================================
def generate_markdown_tables(all_results):
    """Generate markdown tables for thesis sections."""
    md = []

    # Section 6.2.1: Evaluation Metrics Table
    md.append("## Section 6.2.1: Evaluation Metrics\n")
    md.append("| Paradigm | Metric | Mean | Std | Min | Max | n |")
    md.append("|----------|--------|------|-----|-----|-----|---|")

    for paradigm in ['centralized', 'federated']:
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']:
            s = all_results['summary_stats'][paradigm][metric]
            md.append(f"| {paradigm.title()} | {metric.replace('_', ' ').title()} | "
                     f"{s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['n']} |")

    # Section 6.2.2: Baseline Comparison
    md.append("\n## Section 6.2.2: Baseline Comparison\n")
    md.append("| Metric | Baseline (Dec 26) | Centralized (Tuned) | Federated (Tuned) | Cent. Impr. | Fed. Impr. |")
    md.append("|--------|-------------------|---------------------|-------------------|-------------|------------|")

    bc = all_results['baseline_comparison']
    for metric in ['recall', 'auroc', 'f1_score', 'accuracy']:
        base = bc['baseline'][metric]
        cent = bc['centralized_tuned'][metric]
        fed = bc['federated_tuned'][metric]
        cent_imp = bc['improvement_from_baseline'][metric]['centralized_pct']
        fed_imp = bc['improvement_from_baseline'][metric]['federated_pct']
        md.append(f"| {metric.replace('_', ' ').title()} | {base:.4f} | {cent:.4f} | {fed:.4f} | "
                 f"{cent_imp:+.1f}% | {fed_imp:+.1f}% |")

    # Section 6.2.3: Statistical Tests
    md.append("\n## Section 6.2.3: Statistical Significance\n")
    md.append("| Metric | Paired t-test p | Welch p | Cohen's d | Effect Size | 95% CI |")
    md.append("|--------|-----------------|---------|-----------|-------------|--------|")

    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']:
        st = all_results['statistical_tests'][metric]
        p_paired = st['paired_t_test']['p_value']
        p_welch = st['welch_t_test']['p_value']
        d = st['cohens_d']
        effect = st['effect_size']
        ci = st['mean_difference']
        sig = "*" if st['paired_t_test']['significant'] else ""
        md.append(f"| {metric.replace('_', ' ').title()} | {p_paired:.4f}{sig} | {p_welch:.4f} | "
                 f"{d:.3f} | {effect} | [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] |")

    # Section 6.2.3: Coefficient of Variation
    md.append("\n## Section 6.2.3: Coefficient of Variation (Stability)\n")
    md.append("| Metric | Centralized CV% | Federated CV% | More Stable |")
    md.append("|--------|-----------------|---------------|-------------|")

    cv = all_results['coefficient_of_variation']
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']:
        cent_cv = cv['centralized'][metric]
        fed_cv = cv['federated'][metric]
        more_stable = "Centralized" if cent_cv < fed_cv else "Federated"
        md.append(f"| {metric.replace('_', ' ').title()} | {cent_cv:.1f}% | {fed_cv:.1f}% | {more_stable} |")

    # Section 6.4: Fairness Metrics
    md.append("\n## Section 6.4: Fairness & Clinical Risk\n")
    md.append("| Paradigm | Sensitivity | Specificity | FNR | FPR | Clinical Risk |")
    md.append("|----------|-------------|-------------|-----|-----|---------------|")

    fm = all_results['fairness_metrics']
    ra = all_results['clinical_risk_assessment']
    for paradigm in ['centralized', 'federated']:
        agg = fm[paradigm]['aggregate']
        risk = ra[paradigm]
        md.append(f"| {paradigm.title()} | {agg['sensitivity']:.4f} | {agg['specificity']:.4f} | "
                 f"{agg['fnr']:.4f} | {agg['fpr']:.4f} | {risk['overall_assessment']} |")

    # Per-run fairness (federated only - has actual data)
    md.append("\n### Per-Run Fairness Metrics (Federated)\n")
    md.append("| Seed | Sensitivity | Specificity | FNR | FPR |")
    md.append("|------|-------------|-------------|-----|-----|")

    for run in fm['federated']['per_run']:
        md.append(f"| {run['seed']} | {run['sensitivity']:.4f} | {run['specificity']:.4f} | "
                 f"{run['fnr']:.4f} | {run['fpr']:.4f} |")

    return "\n".join(md)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_analysis():
    """Execute complete analysis pipeline."""
    print("=" * 60)
    print("FYP2 Comprehensive Statistical Analysis")
    print("Seeds 44-48 (5 Paired Runs) - Options A + C")
    print("=" * 60)

    # Load data
    print("\n[1/7] Loading data...")
    cent_df, fed_df, baseline, best_run_epochs = load_data()
    print(f"  Centralized: {len(cent_df)} runs (seeds {cent_df['seed'].tolist()})")
    print(f"  Federated: {len(fed_df)} runs (seeds {fed_df['seed'].tolist()})")

    # Option C: Approximate centralized confusion matrix
    print("\n[2/7] Approximating centralized confusion matrix (Option C)...")
    cent_df = add_centralized_confusion_matrix(cent_df)
    print(f"  Added tp_approx, tn_approx, fp_approx, fn_approx columns")

    # Descriptive statistics
    print("\n[3/7] Computing descriptive statistics (Section 6.2.1)...")
    summary_stats = compute_descriptive_stats(cent_df, fed_df)

    # Statistical tests
    print("\n[4/7] Computing statistical tests (Section 6.2.3)...")
    stat_tests = compute_statistical_tests(cent_df, fed_df)

    # Coefficient of variation
    print("\n[5/7] Computing coefficient of variation (Section 6.2.3)...")
    cv_results = compute_coefficient_of_variation(summary_stats)

    # Fairness metrics
    print("\n[6/7] Computing fairness metrics (Section 6.4)...")
    fairness = compute_fairness_metrics(cent_df, fed_df)
    clinical_risk = assess_clinical_risk(fairness)

    # Baseline comparison
    print("\n[7/7] Computing baseline comparison (Section 6.2.2)...")
    baseline_comp = compute_baseline_comparison(cent_df, fed_df, baseline)

    # Compile all results
    all_results = {
        'metadata': {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'seeds_compared': SEEDS_TO_COMPARE,
            'n_runs_per_paradigm': 5,
            'validation_samples': VAL_SAMPLES,
            'val_class_0_normal': VAL_CLASS_0,
            'val_class_1_pneumonia': VAL_CLASS_1,
            'options_used': ['A: Paired metrics comparison', 'C: Approximated centralized confusion matrix']
        },
        'summary_stats': summary_stats,
        'statistical_tests': stat_tests,
        'coefficient_of_variation': cv_results,
        'fairness_metrics': fairness,
        'clinical_risk_assessment': clinical_risk,
        'baseline_comparison': baseline_comp
    }

    # Save JSON
    output_path = BASE_DIR / "statistical_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[OUTPUT] Saved: {output_path}")

    # Generate markdown tables
    md_content = generate_markdown_tables(all_results)
    md_path = BASE_DIR / "ANALYSIS_TABLES.md"
    with open(md_path, 'w') as f:
        f.write(f"# FYP2 Analysis Tables\n\nGenerated: {pd.Timestamp.now()}\n\n")
        f.write(md_content)
    print(f"[OUTPUT] Saved: {md_path}")

    # Save updated dataframes with confusion matrices
    cent_output = BASE_DIR / "centralized" / "experiment_results_with_cm.csv"
    cent_df.to_csv(cent_output, index=False)
    print(f"[OUTPUT] Saved: {cent_output}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n### Key Findings ###\n")

    # Recall comparison
    cent_recall = summary_stats['centralized']['recall']['mean']
    fed_recall = summary_stats['federated']['recall']['mean']
    recall_test = stat_tests['recall']
    print(f"RECALL (Primary Metric):")
    print(f"  Centralized: {cent_recall:.4f} ± {summary_stats['centralized']['recall']['std']:.4f}")
    print(f"  Federated:   {fed_recall:.4f} ± {summary_stats['federated']['recall']['std']:.4f}")
    print(f"  Difference:  {cent_recall - fed_recall:+.4f} (Cent - Fed)")
    print(f"  Paired t-test: p = {recall_test['paired_t_test']['p_value']:.4f} "
          f"{'(significant)' if recall_test['paired_t_test']['significant'] else '(not significant)'}")
    print(f"  Cohen's d:     {recall_test['cohens_d']:.3f} ({recall_test['effect_size']} effect)")

    # Clinical risk
    print(f"\nCLINICAL RISK ASSESSMENT:")
    for paradigm in ['centralized', 'federated']:
        risk = clinical_risk[paradigm]
        print(f"  {paradigm.title()}:")
        print(f"    FNR: {risk['fnr_value']:.4f} ({risk['fnr_risk']} risk)")
        print(f"    FPR: {risk['fpr_value']:.4f} ({risk['fpr_risk']} risk)")
        print(f"    Assessment: {risk['overall_assessment']}")

    # Winner by metric
    print(f"\nWINNER BY METRIC:")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']:
        cent_mean = summary_stats['centralized'][metric]['mean']
        fed_mean = summary_stats['federated'][metric]['mean']
        winner = "Centralized" if cent_mean > fed_mean else "Federated"
        print(f"  {metric.replace('_', ' ').title():12s}: {winner} ({max(cent_mean, fed_mean):.4f} vs {min(cent_mean, fed_mean):.4f})")

    return all_results


if __name__ == "__main__":
    results = run_analysis()
