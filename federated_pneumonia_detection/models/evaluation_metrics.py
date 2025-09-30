from dataclasses import dataclass, asdict
from typing import Any , Dict
import json
@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for binary classification with medical focus."""

    # Basic classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float

    # Medical domain metrics
    sensitivity: float  # Same as recall, but medical terminology
    npv: float  # Negative Predictive Value
    ppv: float  # Positive Predictive Value (same as precision)

    # Advanced metrics
    roc_auc: float
    pr_auc: float
    average_precision: float

    # Confusion matrix components
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Confidence metrics
    mean_confidence: float
    confidence_std: float
    calibration_error: float

    # Sample info
    total_samples: int
    positive_samples: int
    negative_samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'EvaluationMetrics':
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
