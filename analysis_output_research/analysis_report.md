# Comparative Analysis: Federated vs Centralized Learning

**Generated:** 2025-12-23 01:27:43

---

## Executive Summary

This report presents a comparative analysis between **centralized** and **federated**
learning approaches for pneumonia detection from chest X-rays.

### Key Findings

- **Experiments:** 10 centralized runs, 10 federated runs
- **Significant Differences:** 5 of 6 metrics show significant difference (p < 0.05)
- **Accuracy:** Centralized 0.7634 ± 0.0240 vs Federated 0.7127 ± 0.0265
- **Finding:** Federated approach shows 0.0507 lower accuracy

**Conclusion:** Statistically significant differences found in: accuracy, precision, f1, auroc, loss. Further analysis recommended to assess practical significance.

## Methodology

### Experimental Design

- **Comparison Approach:** Sequential execution (centralized first, then federated)
- **Statistical Validity:** 5 runs per approach with different random seeds
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, AUROC

### Centralized Training Configuration
- **Epochs:** 10
- **Batch_Size:** 32
- **Learning_Rate:** 0.001
- **Seed:** 1518965011

### Federated Training Configuration
- **Num_Clients:** 5
- **Num_Rounds:** 5
- **Local_Epochs:** 2
- **Seed:** 1518965011


## Dataset Description

### Overview
- **Total Samples:** 30,227
- **Number of Classes:** 2
- **Class Balance Ratio:** 0.4622

### Class Distribution
- **Class 0:** 20,672 (68.4%)
- **Class 1:** 9,555 (31.6%)


## Results

### Performance Comparison

| Metric | Centralized | Federated | Difference | % Change |
|--------|-------------|-----------|------------|----------|
| Accuracy | 0.7634 ± 0.0240 | 0.7127 ± 0.0265 | -0.0507 | -6.64% |
| Precision | 0.5955 ± 0.0346 | 0.4577 ± 0.0248 | -0.1377 | -23.13% |
| Recall | 0.8052 ± 0.0708 | 0.7507 ± 0.0706 | -0.0545 | -6.77% |
| F1 | 0.6827 ± 0.0320 | 0.5663 ± 0.0129 | -0.1164 | -17.05% |
| Auroc | 0.8409 ± 0.0228 | 0.8055 ± 0.0065 | -0.0354 | -4.21% |
| Loss | 0.6835 ± 0.0591 | 0.5658 ± 0.0506 | -0.1177 | -17.22% |


## Statistical Analysis

### Hypothesis Testing

| Metric | Test Used | p-value | Significant | Cohen's d | Effect |
|--------|-----------|---------|-------------|-----------|--------|
| Accuracy | paired t-test | 0.0013 | **Yes** | -2.005 | Large |
| Precision | paired t-test | <0.0001 | **Yes** | -4.571 | Large |
| Recall | paired t-test | 0.0688 | No | -0.771 | Medium |
| F1 | paired t-test | <0.0001 | **Yes** | -4.776 | Large |
| Auroc | paired t-test | 0.0005 | **Yes** | -2.109 | Large |
| Loss | paired t-test | 0.0008 | **Yes** | -2.140 | Large |

*Note: Significance level α = 0.05*

## Conclusions

### Summary of Findings

The analysis reveals **statistically significant differences** in the following
metrics: Accuracy, Precision, F1, Auroc, Loss.

- **Accuracy:** Federated learning shows significantly lower accuracy compared to centralized learning (paired t-test, diff=-0.0507).
- **Precision:** Federated learning shows significantly lower precision compared to centralized learning (paired t-test, diff=-0.1377).
- **F1:** Federated learning shows significantly lower f1 compared to centralized learning (paired t-test, diff=-0.1164).
- **Auroc:** Federated learning shows significantly lower auroc compared to centralized learning (paired t-test, diff=-0.0354).
- **Loss:** Federated learning shows significantly lower loss compared to centralized learning (paired t-test, diff=-0.1177).

### Implications

1. **Privacy-Performance Trade-off:** Results indicate the feasibility of using
   federated learning for medical image analysis with minimal performance degradation.

2. **Practical Considerations:** The observed differences (if any) should be
   weighed against the privacy benefits of federated learning.

3. **Future Work:** Further investigation with larger datasets and more
   federated clients is recommended.


## Figures

The following visualizations are available in the output directory:

- `auroc_comparison.png`
- `learning_curves_loss.png`
- `learning_curves_val_acc.png`
- `learning_curves_val_f1.png`
- `learning_curves_val_recall.png`
- `metric_comparison.png`
- `metric_difference.png`
- `metric_distributions.png`
- `roc_comparison.png`
