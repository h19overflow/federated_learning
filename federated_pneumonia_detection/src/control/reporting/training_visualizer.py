"""
Advanced visualization and reporting system for training results.
Provides comprehensive training analytics, loss curves, and performance tracking.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader


class TrainingVisualizer:
    """
    Advanced training visualization and reporting system.

    Provides comprehensive analysis of:
    - Training and validation loss curves
    - Metric progression over epochs
    - Learning rate scheduling visualization
    - Training performance analytics
    - Interactive dashboards
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize visualizer with configuration."""
        self.config = ConfigLoader.load_config(config_path) if config_path else {}
        self.constants = SystemConstants()

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_training_report(
        self,
        log_dir: str,
        output_dir: str,
        experiment_name: str = "training_experiment"
    ) -> Dict[str, Any]:
        """
        Create comprehensive training report from TensorBoard logs.

        Args:
            log_dir: Path to TensorBoard log directory
            output_dir: Directory to save visualization outputs
            experiment_name: Name of the experiment

        Returns:
            Dictionary with training statistics and analysis
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Parse TensorBoard logs
        training_data = self._parse_tensorboard_logs(log_dir)

        if not training_data:
            print(f"Warning: No training data found in {log_dir}")
            return {}

        # Create visualizations
        self._create_loss_curves(training_data, output_dir, experiment_name)
        self._create_metrics_dashboard(training_data, output_dir, experiment_name)
        self._create_learning_rate_plot(training_data, output_dir, experiment_name)
        self._create_interactive_dashboard(training_data, output_dir, experiment_name)

        # Generate training statistics
        stats = self._calculate_training_statistics(training_data)

        # Save training report
        report_path = os.path.join(output_dir, f"{experiment_name}_training_report.json")
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Create summary report
        self._create_summary_report(stats, training_data, output_dir, experiment_name)

        return stats

    def _parse_tensorboard_logs(self, log_dir: str) -> Dict[str, Any]:
        """Parse TensorBoard event files to extract training data."""
        training_data = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'epochs': [],
            'steps': []
        }

        try:
            # Find event files
            event_files = []
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if 'tfevents' in file:
                        event_files.append(os.path.join(root, file))

            if not event_files:
                return training_data

            # Process each event file
            for event_file in event_files:
                try:
                    ea = EventAccumulator(event_file)
                    ea.Reload()

                    # Extract scalars
                    scalar_keys = ea.Tags()['scalars']

                    for key in scalar_keys:
                        scalar_events = ea.Scalars(key)
                        for event in scalar_events:
                            step = event.step
                            value = event.value

                            # Map TensorBoard keys to our data structure
                            if 'train_loss' in key.lower() or 'loss/train' in key.lower():
                                training_data['train_loss'].append((step, value))
                            elif 'val_loss' in key.lower() or 'loss/val' in key.lower():
                                training_data['val_loss'].append((step, value))
                            elif 'train_acc' in key.lower() or 'acc/train' in key.lower():
                                training_data['train_acc'].append((step, value))
                            elif 'val_acc' in key.lower() or 'acc/val' in key.lower():
                                training_data['val_acc'].append((step, value))
                            elif 'lr' in key.lower() or 'learning_rate' in key.lower():
                                training_data['learning_rate'].append((step, value))

                except Exception as e:
                    print(f"Error processing {event_file}: {e}")
                    continue

        except Exception as e:
            print(f"Error parsing TensorBoard logs: {e}")

        # Convert to sorted arrays
        for key in training_data:
            if training_data[key]:
                training_data[key] = sorted(training_data[key], key=lambda x: x[0])
                training_data[key] = list(zip(*training_data[key]))

        return training_data

    def _create_loss_curves(
        self,
        training_data: Dict[str, Any],
        output_dir: str,
        experiment_name: str
    ) -> None:
        """Create comprehensive loss curve visualizations."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Analysis - {experiment_name}', fontsize=16, fontweight='bold')

        # Loss curves
        ax = axes[0, 0]
        if training_data['train_loss']:
            steps, values = training_data['train_loss']
            ax.plot(steps, values, label='Training Loss', color='blue', alpha=0.8)
        if training_data['val_loss']:
            steps, values = training_data['val_loss']
            ax.plot(steps, values, label='Validation Loss', color='red', alpha=0.8)

        ax.set_title('Training & Validation Loss')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy curves
        ax = axes[0, 1]
        if training_data['train_acc']:
            steps, values = training_data['train_acc']
            ax.plot(steps, values, label='Training Accuracy', color='green', alpha=0.8)
        if training_data['val_acc']:
            steps, values = training_data['val_acc']
            ax.plot(steps, values, label='Validation Accuracy', color='orange', alpha=0.8)

        ax.set_title('Training & Validation Accuracy')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate schedule
        ax = axes[1, 0]
        if training_data['learning_rate']:
            steps, values = training_data['learning_rate']
            ax.plot(steps, values, label='Learning Rate', color='purple', alpha=0.8)
            ax.set_yscale('log')

        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss smoothing comparison
        ax = axes[1, 1]
        if training_data['train_loss']:
            steps, values = training_data['train_loss']
            if len(values) > 10:
                # Original loss
                ax.plot(steps, values, alpha=0.3, color='blue', label='Raw Training Loss')
                # Smoothed loss (moving average)
                window_size = min(50, len(values) // 10)
                if window_size > 1:
                    smoothed = self._moving_average(values, window_size)
                    ax.plot(steps[:len(smoothed)], smoothed, color='blue', linewidth=2, label='Smoothed Training Loss')

        ax.set_title('Loss Smoothing Analysis')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{experiment_name}_training_curves.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_metrics_dashboard(
        self,
        training_data: Dict[str, Any],
        output_dir: str,
        experiment_name: str
    ) -> None:
        """Create comprehensive metrics dashboard."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Metrics Dashboard - {experiment_name}', fontsize=16, fontweight='bold')

        # 1. Loss comparison
        ax = axes[0, 0]
        if training_data['train_loss'] and training_data['val_loss']:
            train_steps, train_loss = training_data['train_loss']
            val_steps, val_loss = training_data['val_loss']

            # Calculate overfitting indicator
            if len(train_loss) > 10 and len(val_loss) > 10:
                train_final = np.mean(train_loss[-10:])
                val_final = np.mean(val_loss[-10:])
                overfitting_gap = val_final - train_final

                ax.plot(train_steps, train_loss, label=f'Train (final: {train_final:.3f})', alpha=0.8)
                ax.plot(val_steps, val_loss, label=f'Val (final: {val_final:.3f})', alpha=0.8)
                ax.set_title(f'Loss Comparison\nOverfitting Gap: {overfitting_gap:.3f}')

        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Convergence analysis
        ax = axes[0, 1]
        if training_data['val_loss']:
            steps, values = training_data['val_loss']
            if len(values) > 20:
                # Calculate convergence indicator (variance of recent values)
                recent_values = values[-20:]
                convergence_variance = np.var(recent_values)

                ax.plot(steps, values, alpha=0.8)
                ax.axhline(y=np.mean(recent_values), color='red', linestyle='--',
                          label=f'Recent Mean: {np.mean(recent_values):.3f}')
                ax.set_title(f'Convergence Analysis\nRecent Variance: {convergence_variance:.4f}')

        ax.set_xlabel('Steps')
        ax.set_ylabel('Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Training speed analysis
        ax = axes[0, 2]
        if training_data['train_loss']:
            steps, values = training_data['train_loss']
            if len(steps) > 1:
                # Calculate training speed (steps per unit time - approximated)
                step_intervals = np.diff(steps)
                avg_interval = np.mean(step_intervals) if len(step_intervals) > 0 else 1

                ax.hist(step_intervals, bins=30, alpha=0.7, edgecolor='black')
                ax.axvline(x=avg_interval, color='red', linestyle='--',
                          label=f'Avg Interval: {avg_interval:.1f}')
                ax.set_title('Training Step Intervals')

        ax.set_xlabel('Steps Between Updates')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Learning rate impact
        ax = axes[1, 0]
        if training_data['learning_rate'] and training_data['train_loss']:
            lr_steps, lr_values = training_data['learning_rate']
            loss_steps, loss_values = training_data['train_loss']

            # Create secondary y-axis
            ax2 = ax.twinx()

            line1 = ax.plot(lr_steps, lr_values, 'g-', alpha=0.8, label='Learning Rate')
            line2 = ax2.plot(loss_steps, loss_values, 'b-', alpha=0.8, label='Training Loss')

            ax.set_xlabel('Steps')
            ax.set_ylabel('Learning Rate', color='g')
            ax2.set_ylabel('Training Loss', color='b')
            ax.set_title('Learning Rate vs Training Loss')

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')

        # 5. Training efficiency
        ax = axes[1, 1]
        if training_data['train_loss']:
            steps, values = training_data['train_loss']
            if len(values) > 10:
                # Calculate improvement rate
                improvement_rates = []
                window = 10
                for i in range(window, len(values)):
                    old_avg = np.mean(values[i-window:i])
                    new_avg = np.mean(values[i-window//2:i])
                    improvement_rate = (old_avg - new_avg) / old_avg if old_avg != 0 else 0
                    improvement_rates.append(improvement_rate)

                if improvement_rates:
                    ax.plot(steps[window:len(improvement_rates)+window], improvement_rates, alpha=0.8)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    ax.set_title('Training Improvement Rate')

        ax.set_xlabel('Steps')
        ax.set_ylabel('Improvement Rate')
        ax.grid(True, alpha=0.3)

        # 6. Final metrics summary
        ax = axes[1, 2]
        summary_metrics = self._calculate_training_statistics(training_data)

        # Create text summary
        summary_text = ""
        if 'final_train_loss' in summary_metrics:
            summary_text += f"Final Train Loss: {summary_metrics['final_train_loss']:.3f}\n"
        if 'final_val_loss' in summary_metrics:
            summary_text += f"Final Val Loss: {summary_metrics['final_val_loss']:.3f}\n"
        if 'best_val_loss' in summary_metrics:
            summary_text += f"Best Val Loss: {summary_metrics['best_val_loss']:.3f}\n"
        if 'total_steps' in summary_metrics:
            summary_text += f"Total Steps: {summary_metrics['total_steps']}\n"
        if 'convergence_epoch' in summary_metrics:
            summary_text += f"Convergence Step: {summary_metrics['convergence_epoch']}\n"

        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Training Summary')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{experiment_name}_metrics_dashboard.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_learning_rate_plot(
        self,
        training_data: Dict[str, Any],
        output_dir: str,
        experiment_name: str
    ) -> None:
        """Create detailed learning rate analysis."""

        if not training_data['learning_rate']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Learning Rate Analysis - {experiment_name}', fontsize=16, fontweight='bold')

        steps, lr_values = training_data['learning_rate']

        # 1. Learning rate schedule
        ax = axes[0, 0]
        ax.plot(steps, lr_values, linewidth=2, color='purple')
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)

        # 2. Log scale learning rate
        ax = axes[0, 1]
        ax.plot(steps, lr_values, linewidth=2, color='purple')
        ax.set_yscale('log')
        ax.set_title('Learning Rate (Log Scale)')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate (log)')
        ax.grid(True, alpha=0.3)

        # 3. Learning rate changes
        ax = axes[1, 0]
        if len(lr_values) > 1:
            lr_changes = np.diff(lr_values)
            ax.plot(steps[1:], lr_changes, alpha=0.8, color='orange')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Learning Rate Changes')
            ax.set_xlabel('Steps')
            ax.set_ylabel('LR Change')
            ax.grid(True, alpha=0.3)

        # 4. Learning rate statistics
        ax = axes[1, 1]
        lr_stats = {
            'Min LR': np.min(lr_values),
            'Max LR': np.max(lr_values),
            'Mean LR': np.mean(lr_values),
            'Final LR': lr_values[-1] if lr_values else 0,
            'LR Reductions': np.sum(np.diff(lr_values) < -1e-10) if len(lr_values) > 1 else 0
        }

        stats_text = "\n".join([f"{k}: {v:.2e}" for k, v in lr_stats.items()])
        ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Learning Rate Statistics')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{experiment_name}_learning_rate.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_interactive_dashboard(
        self,
        training_data: Dict[str, Any],
        output_dir: str,
        experiment_name: str
    ) -> None:
        """Create interactive Plotly dashboard."""

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Loss Curves', 'Accuracy Curves', 'Learning Rate', 'Training Progress'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Loss curves
        if training_data['train_loss']:
            steps, values = training_data['train_loss']
            fig.add_trace(
                go.Scatter(x=steps, y=values, name='Training Loss',
                          line=dict(color='blue'), opacity=0.8),
                row=1, col=1
            )

        if training_data['val_loss']:
            steps, values = training_data['val_loss']
            fig.add_trace(
                go.Scatter(x=steps, y=values, name='Validation Loss',
                          line=dict(color='red'), opacity=0.8),
                row=1, col=1
            )

        # Accuracy curves
        if training_data['train_acc']:
            steps, values = training_data['train_acc']
            fig.add_trace(
                go.Scatter(x=steps, y=values, name='Training Accuracy',
                          line=dict(color='green'), opacity=0.8),
                row=1, col=2
            )

        if training_data['val_acc']:
            steps, values = training_data['val_acc']
            fig.add_trace(
                go.Scatter(x=steps, y=values, name='Validation Accuracy',
                          line=dict(color='orange'), opacity=0.8),
                row=1, col=2
            )

        # Learning rate
        if training_data['learning_rate']:
            steps, values = training_data['learning_rate']
            fig.add_trace(
                go.Scatter(x=steps, y=values, name='Learning Rate',
                          line=dict(color='purple'), opacity=0.8),
                row=2, col=1
            )

        # Training progress (combined view)
        if training_data['train_loss'] and training_data['val_loss']:
            train_steps, train_values = training_data['train_loss']
            val_steps, val_values = training_data['val_loss']

            # Normalize values for combined view
            train_norm = np.array(train_values) / np.max(train_values) if train_values else []
            val_norm = np.array(val_values) / np.max(val_values) if val_values else []

            fig.add_trace(
                go.Scatter(x=train_steps, y=train_norm, name='Train Loss (norm)',
                          line=dict(color='blue', dash='dot'), opacity=0.6),
                row=2, col=2
            )

            fig.add_trace(
                go.Scatter(x=val_steps, y=val_norm, name='Val Loss (norm)',
                          line=dict(color='red', dash='dot'), opacity=0.6),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title=f'Interactive Training Dashboard - {experiment_name}',
            showlegend=True,
            height=800
        )

        # Save interactive plot
        pyo.plot(fig, filename=os.path.join(output_dir, f"{experiment_name}_interactive_dashboard.html"),
                auto_open=False)

    def _calculate_training_statistics(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive training statistics."""

        stats = {}

        # Loss statistics
        if training_data['train_loss']:
            _, train_loss = training_data['train_loss']
            stats['final_train_loss'] = train_loss[-1] if train_loss else None
            stats['min_train_loss'] = np.min(train_loss) if train_loss else None
            stats['max_train_loss'] = np.max(train_loss) if train_loss else None

        if training_data['val_loss']:
            _, val_loss = training_data['val_loss']
            stats['final_val_loss'] = val_loss[-1] if val_loss else None
            stats['best_val_loss'] = np.min(val_loss) if val_loss else None
            stats['best_val_loss_step'] = np.argmin(val_loss) if val_loss else None

        # Accuracy statistics
        if training_data['train_acc']:
            _, train_acc = training_data['train_acc']
            stats['final_train_acc'] = train_acc[-1] if train_acc else None
            stats['best_train_acc'] = np.max(train_acc) if train_acc else None

        if training_data['val_acc']:
            _, val_acc = training_data['val_acc']
            stats['final_val_acc'] = val_acc[-1] if val_acc else None
            stats['best_val_acc'] = np.max(val_acc) if val_acc else None
            stats['best_val_acc_step'] = np.argmax(val_acc) if val_acc else None

        # Training progress statistics
        if training_data['train_loss']:
            steps, _ = training_data['train_loss']
            stats['total_steps'] = steps[-1] if steps else 0
            stats['training_duration_steps'] = len(steps)

        # Convergence analysis
        if training_data['val_loss']:
            _, val_loss = training_data['val_loss']
            if len(val_loss) > 20:
                # Find convergence point (where variance becomes small)
                window_size = min(10, len(val_loss) // 4)
                variances = []
                for i in range(window_size, len(val_loss)):
                    window_values = val_loss[i-window_size:i]
                    variances.append(np.var(window_values))

                if variances:
                    convergence_threshold = np.mean(variances) * 0.1  # 10% of mean variance
                    convergence_points = [i for i, var in enumerate(variances) if var < convergence_threshold]
                    if convergence_points:
                        stats['convergence_epoch'] = convergence_points[0] + window_size

        # Learning rate statistics
        if training_data['learning_rate']:
            _, lr_values = training_data['learning_rate']
            stats['initial_lr'] = lr_values[0] if lr_values else None
            stats['final_lr'] = lr_values[-1] if lr_values else None
            stats['min_lr'] = np.min(lr_values) if lr_values else None
            stats['max_lr'] = np.max(lr_values) if lr_values else None

            if len(lr_values) > 1:
                lr_reductions = np.sum(np.diff(lr_values) < -1e-10)
                stats['lr_reductions'] = int(lr_reductions)

        return stats

    def _create_summary_report(
        self,
        stats: Dict[str, Any],
        training_data: Dict[str, Any],
        output_dir: str,
        experiment_name: str
    ) -> None:
        """Create comprehensive summary report."""

        report_path = os.path.join(output_dir, f"{experiment_name}_training_summary.txt")

        with open(report_path, 'w') as f:
            f.write(f"Training Summary Report - {experiment_name}\n")
            f.write("=" * 60 + "\n\n")

            # Training Overview
            f.write("Training Overview:\n")
            f.write("-" * 20 + "\n")
            if 'total_steps' in stats:
                f.write(f"Total Training Steps: {stats['total_steps']}\n")
            if 'training_duration_steps' in stats:
                f.write(f"Training Duration (steps): {stats['training_duration_steps']}\n")
            if 'convergence_epoch' in stats:
                f.write(f"Convergence Step: {stats['convergence_epoch']}\n")
            f.write("\n")

            # Loss Analysis
            f.write("Loss Analysis:\n")
            f.write("-" * 15 + "\n")
            if 'final_train_loss' in stats:
                f.write(f"Final Training Loss: {stats['final_train_loss']:.4f}\n")
            if 'final_val_loss' in stats:
                f.write(f"Final Validation Loss: {stats['final_val_loss']:.4f}\n")
            if 'best_val_loss' in stats:
                f.write(f"Best Validation Loss: {stats['best_val_loss']:.4f}\n")
            if 'best_val_loss_step' in stats:
                f.write(f"Best Loss at Step: {stats['best_val_loss_step']}\n")
            f.write("\n")

            # Accuracy Analysis
            f.write("Accuracy Analysis:\n")
            f.write("-" * 18 + "\n")
            if 'final_train_acc' in stats:
                f.write(f"Final Training Accuracy: {stats['final_train_acc']:.4f}\n")
            if 'final_val_acc' in stats:
                f.write(f"Final Validation Accuracy: {stats['final_val_acc']:.4f}\n")
            if 'best_val_acc' in stats:
                f.write(f"Best Validation Accuracy: {stats['best_val_acc']:.4f}\n")
            if 'best_val_acc_step' in stats:
                f.write(f"Best Accuracy at Step: {stats['best_val_acc_step']}\n")
            f.write("\n")

            # Learning Rate Analysis
            f.write("Learning Rate Analysis:\n")
            f.write("-" * 23 + "\n")
            if 'initial_lr' in stats:
                f.write(f"Initial Learning Rate: {stats['initial_lr']:.2e}\n")
            if 'final_lr' in stats:
                f.write(f"Final Learning Rate: {stats['final_lr']:.2e}\n")
            if 'lr_reductions' in stats:
                f.write(f"Learning Rate Reductions: {stats['lr_reductions']}\n")
            f.write("\n")

            # Training Quality Assessment
            f.write("Training Quality Assessment:\n")
            f.write("-" * 28 + "\n")

            # Overfitting check
            if 'final_train_loss' in stats and 'final_val_loss' in stats:
                overfitting_gap = stats['final_val_loss'] - stats['final_train_loss']
                f.write(f"Overfitting Gap (Val - Train): {overfitting_gap:.4f}\n")
                if overfitting_gap > 0.1:
                    f.write("⚠️  WARNING: Possible overfitting detected\n")
                else:
                    f.write("✅ Good generalization (low overfitting gap)\n")

            # Convergence check
            if 'convergence_epoch' in stats and 'total_steps' in stats:
                convergence_ratio = stats['convergence_epoch'] / stats['total_steps']
                f.write(f"Convergence Ratio: {convergence_ratio:.2f}\n")
                if convergence_ratio < 0.5:
                    f.write("✅ Fast convergence achieved\n")
                elif convergence_ratio < 0.8:
                    f.write("➡️  Normal convergence speed\n")
                else:
                    f.write("⚠️  Slow convergence - consider adjusting hyperparameters\n")

            f.write("\n")

            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 15 + "\n")

            if 'final_val_loss' in stats and 'best_val_loss' in stats:
                if stats['final_val_loss'] > stats['best_val_loss'] * 1.05:
                    f.write("• Consider early stopping to prevent overfitting\n")
                    f.write("• Validation loss increased from best - training may have continued too long\n")

            if 'lr_reductions' in stats:
                if stats['lr_reductions'] == 0:
                    f.write("• No learning rate reductions occurred - consider LR scheduling\n")
                elif stats['lr_reductions'] > 5:
                    f.write("• Many LR reductions - consider starting with lower initial LR\n")

            if 'final_train_acc' in stats and 'final_val_acc' in stats:
                acc_gap = stats['final_train_acc'] - stats['final_val_acc']
                if acc_gap > 0.1:
                    f.write("• Large accuracy gap suggests overfitting - consider regularization\n")

    @staticmethod
    def _moving_average(data: List[float], window_size: int) -> List[float]:
        """Calculate moving average for smoothing."""
        if window_size <= 1:
            return data

        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window = data[start_idx:i + 1]
            smoothed.append(np.mean(window))

        return smoothed
