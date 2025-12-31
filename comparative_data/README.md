# Comparative Data - Centralized vs Federated Learning

Consolidated results for comparative analysis of pneumonia detection models.

---

## Directory Structure

```
comparative_data/
├── README.md (this file)
├── AGGREGATION_CONFIRMED.md (how results are collected)
├── centralized/ (10 complete runs, seeds 42-51)
│   ├── experiment_results.csv (all 10 centralized runs)
│   ├── statistical_analysis.json (centralized summary stats)
│   ├── best_run_epochs.csv (epoch-level metrics from best run)
│   ├── best_run_metadata.json (best run metadata)
│   └── centralized_summary.json (complete training summary)
├── federated/ (awaiting completion)
│   ├── METRICS_EXTRACTION.md (current status & what's missing)
│   ├── federated_pneumonia_detection.csv (latest training session epochs)
│   ├── federated_pneumonia_detection_metadata_*.json (4 client sessions)
│   └── federated_summary.json (latest session complete metrics)
├── data_validation/
│   ├── validation_results.json (data quality checks)
│   ├── DATA_VALIDATION_REPORT.md (full validation report)
│   └── images/ (8 visualization files)
├── baseline/
│   └── pre_tuning_federated.json (Dec 26 results for tuning comparison)
└── metadata/
    ├── current_config.yaml (experiment configuration)
    └── experiment_run.log (execution log)
```

---

## Current Status

### ✅ Centralized Experiments: COMPLETE

| Metric | Mean ± Std | Min | Max | N |
|--------|------------|-----|-----|---|
| **Recall** | **0.863 ± 0.037** | 0.785 | 0.917 | 10 |
| **AUROC** | **0.889 ± 0.008** | 0.878 | 0.901 | 10 |
| Accuracy | 0.778 ± 0.028 | 0.722 | 0.809 | 10 |
| Precision | 0.608 ± 0.041 | 0.536 | 0.668 | 10 |
| F1-Score | 0.711 ± 0.019 | 0.676 | 0.735 | 10 |

**Best Run:** Seed 48, Recall = 0.917, AUROC = 0.898

### ⏳ Federated Experiments: IN PROGRESS

**Expected:** 10 runs (seeds 42-51)
**Current:** Running (check `experiment_run.log` for progress)
**ETA:** ~47 hours total (if sequential) or less if parallelized

**Partial Results Available:**
- 4 client training sessions recorded
- Best observed recall: 0.9214 (client-level)
- Awaiting server-side aggregated metrics

---

## Data Validation Summary

**Total Samples:** 30,227
- Normal: 20,672 (68.4%)
- Pneumonia: 9,555 (31.6%)

**Splits:**
- Training: 24,181 (80%)
- Validation: 6,046 (20%)
- Stratified: Yes

**Quality:**
- Missing values: 0
- Image size: 1024×1024 PNG (all uniform)
- Preprocessing: Validated

---

## Files for Section 6.2-6.6 Analysis

### 6.2.1 Evaluation Metrics
- Primary: `centralized/experiment_results.csv`
- Detailed: `centralized/best_run_epochs.csv`
- Summary: `centralized/centralized_summary.json`

### 6.2.2 Baseline Comparison
- Current centralized: `centralized/statistical_analysis.json`
- Current federated: `federated/METRICS_EXTRACTION.md` (status)
- Historical: `baseline/pre_tuning_federated.json`

### 6.2.3 Cross-Validation (Variance Analysis)
- Centralized: `centralized/statistical_analysis.json`
- Coefficient of Variation:
  - Recall: 4.3% (very stable)
  - AUROC: 0.9% (extremely stable)
  - Accuracy: 3.6%

### 6.3 Hyperparameter Tuning
- Current config: `metadata/current_config.yaml`
- Baseline (before tuning): `baseline/pre_tuning_federated.json`

### 6.4 Bias & Fairness
- Data distribution: `data_validation/validation_results.json`
- Visualizations: `data_validation/images/`

### 6.5 System Testing
- Execution log: `metadata/experiment_run.log`
- Configuration: `metadata/current_config.yaml`

### 6.6 Acceptance Testing
- To be created after federated completion

---

## Quick Status Check

```powershell
# Check if federated experiments completed
(Get-Content "../analysis/experiment_results.csv").Count
# Expected: 21 when complete (1 header + 10 centralized + 10 federated)

# Count federated runs
(Select-String -Path "../analysis/experiment_results.csv" -Pattern "federated").Count
# Expected: 10 when complete

# Check for results files
(Get-ChildItem "../results_analysis_*.json").Count
# Expected: 10 when complete
```

---

## Next Steps (After Federated Completion)

### Phase 1: Verify Data
- [ ] Confirm `analysis/experiment_results.csv` has 21 rows
- [ ] Verify 10 `results_analysis_*.json` files exist
- [ ] Check for missing values in critical columns

### Phase 2: Update Comparative Data
```bash
# Copy completed federated results
cp ../analysis/experiment_results.csv centralized/
cp ../analysis/statistical_analysis.json centralized/
cp ../results_analysis_*.json federated/
```

### Phase 3: Generate Comparative Analysis
- [ ] Update `statistical_analysis.json` with both paradigms
- [ ] Create comparative visualizations
- [ ] Generate t-test results
- [ ] Calculate effect sizes
- [ ] Create confusion matrix comparisons

### Phase 4: Create Final Report
- [ ] Section 6.2: Model Evaluation
- [ ] Section 6.3: Hyperparameter Tuning
- [ ] Section 6.4: Bias & Fairness
- [ ] Section 6.5: System Testing
- [ ] Section 6.6: Acceptance Testing

---

## Key Findings (Centralized Only - So Far)

### Strengths:
✅ High recall (86.3%) - Good for medical screening
✅ Low variance across seeds (CV = 4.3%)
✅ Consistent AUROC (0.889 ± 0.008)
✅ Fast training (43-77 minutes on GPU)

### Areas for Improvement:
⚠️ Moderate precision (60.8%) - Some false positives
⚠️ Class imbalance (2.16:1 ratio)

### Clinical Implications:
- Sensitivity: 86.3% (good - catches most pneumonia cases)
- Specificity: TBD (need confusion matrix extraction)
- False Negative Rate: TBD (critical for medical safety)

---

## References

- **CLAUDE.md**: Full project documentation at project root
- **AGGREGATION_CONFIRMED.md**: How automatic aggregation works
- **METRICS_EXTRACTION.md**: Federated experiment status
- **Data Validation Report**: `data_validation/DATA_VALIDATION_REPORT.md`

---

## Last Updated
2025-12-29 - Awaiting federated experiment completion
