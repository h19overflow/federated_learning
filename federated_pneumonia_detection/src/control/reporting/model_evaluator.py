"""
Comprehensive model evaluation and metrics system for pneumonia detection.
Provides detailed post-training analysis with medical domain-specific metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve
import pytorch_lightning as pl
from federated_pneumonia_detection.src.utils.logger import get_logger
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.models.evaluation_metrics import  EvaluationMetrics

class ModelEvaluator:
    """
    Comprehensive model evaluation system with medical domain focus.

    Provides detailed analysis including:
    - Standard classification metrics
    - Medical domain specific metrics (sensitivity, specificity, PPV, NPV)
    - ROC and PR curves
    - Confidence calibration analysis
    - Detailed visualizations
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize evaluator with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = ConfigLoader.load_config(config_path) if config_path else {}
        self.constants = SystemConstants()

    def evaluate_model(
        self,
        model: pl.LightningModule,
        dataloader: DataLoader,
        output_dir: str,
        model_name: str = "model"
    ) -> EvaluationMetrics:
        """
        Comprehensive model evaluation with detailed analysis.

        Args:
            model: Trained PyTorch Lightning model
            dataloader: DataLoader for evaluation data
            output_dir: Directory to save evaluation results
            model_name: Name identifier for the model

        Returns:
            EvaluationMetrics: Comprehensive evaluation metrics
        """
        # Ensure output directory exists
        self.logger.info(f"Creating output directory: {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set model to evaluation mode
        self.logger.info(f"Setting model to evaluation mode")
        model.eval()

        # Collect predictions and ground truth
        self.logger.info(f"Collecting predictions and ground truth")
        all_predictions = []
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                self.logger.info(f"Processing batch: {images.shape}, {labels.shape}")
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                # Get model outputs
                self.logger.info(f"Getting model outputs")
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
                predictions = (probabilities > 0.5).int()

                self.logger.info(f"Predictions: {predictions.shape}, Probabilities: {probabilities.shape}")
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        self.logger.info(f"Converting to numpy arrays")
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)

        # Calculate comprehensive metrics
        self.logger.info(f"Calculating comprehensive metrics")
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)

        # Generate visualizations
        self.logger.info(f"Generating visualizations")
        self._create_visualizations(y_true, y_pred, y_prob, output_dir, model_name)

        # Save detailed report
        self.logger.info(f"Saving detailed report")
        self._save_detailed_report(metrics, y_true, y_pred, output_dir, model_name)

        # Save metrics
        self.logger.info(f"Saving metrics")
        metrics.to_json(os.path.join(output_dir, f"{model_name}_metrics.json"))

        return metrics

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics."""

        # Confusion matrix components
        self.logger.info(f"Calculating confusion matrix components")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Basic metrics
        self.logger.info(f"Calculating basic metrics")
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Medical domain metrics
        self.logger.info(f"Calculating medical domain metrics")
        sensitivity = recall  # Same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = precision  # Same as precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Advanced metrics
        self.logger.info(f"Calculating advanced metrics")
        roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        avg_precision = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

        # Confidence metrics
        self.logger.info(f"Calculating confidence metrics")
        mean_confidence = float(np.mean(y_prob))
        confidence_std = float(np.std(y_prob))

        # Calibration error (Brier score approximation)
        self.logger.info(f"Calculating calibration error")
        calibration_error = float(np.mean((y_prob - y_true) ** 2))

        # Sample information
        self.logger.info(f"Calculating sample information")
        total_samples = len(y_true)
        positive_samples = np.sum(y_true)
        negative_samples = total_samples - positive_samples

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            sensitivity=sensitivity,
            npv=npv,
            ppv=ppv,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            average_precision=avg_precision,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            mean_confidence=mean_confidence,
            confidence_std=confidence_std,
            calibration_error=calibration_error,
            total_samples=total_samples,
            positive_samples=int(positive_samples),
            negative_samples=int(negative_samples)
        )

    def _create_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        output_dir: str,
        model_name: str
    ) -> None:
        """Create comprehensive visualization plots."""

        # Set style
        self.logger.info(f"Setting style")
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Evaluation Dashboard - {model_name}', fontsize=16, fontweight='bold')

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # 2. ROC Curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            axes[0, 1].plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            axes[0, 2].plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
            axes[0, 2].set_xlabel('Recall')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].set_title('Precision-Recall Curve')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Confidence Distribution
        axes[1, 0].hist(y_prob[y_true == 0], alpha=0.7, label='Negative', bins=30, color='red')
        axes[1, 0].hist(y_prob[y_true == 1], alpha=0.7, label='Positive', bins=30, color='blue')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Calibration Plot
        if len(np.unique(y_true)) > 1:
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
            axes[1, 1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            axes[1, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            axes[1, 1].set_xlabel('Mean Predicted Probability')
            axes[1, 1].set_ylabel('Fraction of Positives')
            axes[1, 1].set_title('Calibration Plot')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Metrics Summary
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        metrics_text = f"""
        Accuracy: {metrics.accuracy:.3f}
        Precision: {metrics.precision:.3f}
        Recall: {metrics.recall:.3f}
        F1-Score: {metrics.f1_score:.3f}
        Specificity: {metrics.specificity:.3f}
        ROC AUC: {metrics.roc_auc:.3f}
        PR AUC: {metrics.pr_auc:.3f}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('Metrics Summary')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_evaluation_dashboard.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _save_detailed_report(
        self,
        metrics: EvaluationMetrics,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dir: str,
        model_name: str
    ) -> None:
        """Save detailed evaluation report."""

        report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.txt")

        with open(report_path, 'w') as f:
            f.write(f"Detailed Evaluation Report - {model_name}\n")
            f.write("=" * 50 + "\n\n")

            # Basic Information
            f.write("Dataset Information:\n")
            f.write(f"Total Samples: {metrics.total_samples}\n")
            f.write(f"Positive Samples: {metrics.positive_samples} ({metrics.positive_samples/metrics.total_samples:.1%})\n")
            f.write(f"Negative Samples: {metrics.negative_samples} ({metrics.negative_samples/metrics.total_samples:.1%})\n\n")

            # Classification Metrics
            f.write("Classification Metrics:\n")
            f.write(f"Accuracy: {metrics.accuracy:.4f}\n")
            f.write(f"Precision (PPV): {metrics.precision:.4f}\n")
            f.write(f"Recall (Sensitivity): {metrics.recall:.4f}\n")
            f.write(f"F1-Score: {metrics.f1_score:.4f}\n")
            f.write(f"Specificity: {metrics.specificity:.4f}\n")
            f.write(f"NPV: {metrics.npv:.4f}\n\n")

            # Advanced Metrics
            f.write("Advanced Metrics:\n")
            f.write(f"ROC AUC: {metrics.roc_auc:.4f}\n")
            f.write(f"PR AUC: {metrics.pr_auc:.4f}\n")
            f.write(f"Average Precision: {metrics.average_precision:.4f}\n\n")

            # Confusion Matrix
            f.write("Confusion Matrix:\n")
            f.write(f"True Positives: {metrics.true_positives}\n")
            f.write(f"True Negatives: {metrics.true_negatives}\n")
            f.write(f"False Positives: {metrics.false_positives}\n")
            f.write(f"False Negatives: {metrics.false_negatives}\n\n")

            # Confidence Analysis
            f.write("Confidence Analysis:\n")
            f.write(f"Mean Confidence: {metrics.mean_confidence:.4f}\n")
            f.write(f"Confidence Std: {metrics.confidence_std:.4f}\n")
            f.write(f"Calibration Error: {metrics.calibration_error:.4f}\n\n")

            # Medical Interpretation
            f.write("Medical Interpretation:\n")
            f.write("- High Sensitivity (Recall) is crucial for pneumonia detection\n")
            f.write("- High Specificity reduces false alarms\n")
            f.write("- PPV indicates reliability of positive predictions\n")
            f.write("- NPV indicates reliability of negative predictions\n")

            # Sklearn Classification Report
            f.write("\nDetailed Classification Report:\n")
            f.write(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

    def compare_models(
        self,
        metrics_list: List[EvaluationMetrics],
        model_names: List[str],
        output_dir: str
    ) -> pd.DataFrame:
        """
        Compare multiple models and create comparison visualizations.

        Args:
            metrics_list: List of EvaluationMetrics for each model
            model_names: List of model names
            output_dir: Directory to save comparison results

        Returns:
            DataFrame with comparison results
        """
        # Create comparison DataFrame
        comparison_data = []
        for metrics, name in zip(metrics_list, model_names):
            data = metrics.to_dict()
            data['model_name'] = name
            comparison_data.append(data)

        df_comparison = pd.DataFrame(comparison_data)

        # Save comparison table
        comparison_path = os.path.join(output_dir, "model_comparison.csv")
        df_comparison.to_csv(comparison_path, index=False)

        # Create comparison visualizations
        self._create_comparison_plots(df_comparison, output_dir)

        return df_comparison

    def _create_comparison_plots(self, df: pd.DataFrame, output_dir: str) -> None:
        """Create comparison visualization plots."""

        # Key metrics for comparison
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']

        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')

        for i, metric in enumerate(key_metrics):
            row, col = i // 3, i % 3
            ax = axes[row, col]

            bars = ax.bar(df['model_name'], df[metric], alpha=0.8)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')

            # Rotate x-axis labels if needed
            if len(df['model_name'].iloc[0]) > 8:
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()
