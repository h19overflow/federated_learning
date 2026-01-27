"""Text report exporter for formatted summaries."""

from datetime import datetime
from typing import Any, Dict

from .base import DataExporter


class TextReportExporter(DataExporter):
    """Exports formatted text summary report."""

    def export(self, data: Dict[str, Any]) -> str:
        """Generate formatted text report."""
        lines = ["=" * 80, "TRAINING RUN SUMMARY REPORT", "=" * 80, ""]
        metadata = data.get("metadata", {})
        lines += ["EXPERIMENT INFORMATION", "-" * 80]
        lines.append(f"Experiment: {metadata.get('experiment_name', 'N/A')}")
        lines.append(
            f"Status: {data.get('status', 'N/A')} | Epochs: {metadata.get('total_epochs', 'N/A')}",  # noqa: E501
        )
        final = data.get("final_metrics", {})
        lines += ["", "FINAL METRICS", "-" * 80]
        for k, v in [
            ("Acc", "accuracy"),
            ("Recall", "recall"),
            ("F1", "f1_score"),
            ("AUC", "auc"),
        ]:
            val = final.get(v, 0)
            lines.append(f"{k:<8} {val:.4f} ({val * 100:.1f}%)")
        history = data.get("training_history", [])
        if history:
            lines += ["", "HISTORY", "-" * 80]
            for e in history[:5]:
                lines.append(
                    f"E{e.get('epoch', 1):<2} | TL:{e.get('train_loss', 0):.3f} "
                    f"VL:{e.get('val_loss', 0):.3f} VA:{e.get('val_acc', 0) * 100:.1f}%",  # noqa: E501
                )
        lines += [
            "",
            "=" * 80,
            f"Report: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
        ]
        return "\n".join(lines)

    def get_media_type(self) -> str:
        return "text/plain"

    def get_file_extension(self) -> str:
        return "txt"
