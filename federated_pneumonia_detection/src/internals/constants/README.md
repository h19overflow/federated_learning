# Constants Module

## Purpose
Centralized location for global constants and metric names used across the backend to prevent magic strings and ensure consistency.

## Key Files
- `metric_names.py`: Defines standard keys for training metrics (e.g., `VAL_ACCURACY`, `TRAIN_LOSS`) and confusion matrix components.

## Usage
Import constants to ensure consistency when logging or retrieving metrics:

```python
from federated_pneumonia_detection.src.internals.constants import VAL_ACCURACY, TRAIN_LOSS

# In training loop
self.log(VAL_ACCURACY, accuracy)
```

## Metric Names Reference
| Constant | Value | Description |
|----------|-------|-------------|
| `VAL_LOSS` | `"val_loss"` | Validation loss |
| `VAL_ACCURACY` | `"val_accuracy"` | Validation accuracy |
| `TRAIN_LOSS` | `"train_loss"` | Training loss |
| `TRAIN_ACCURACY` | `"train_accuracy"` | Training accuracy |
| `SENSITIVITY` | `"sensitivity"` | Recall / True Positive Rate |
| `SPECIFICITY` | `"specificity"` | True Negative Rate |
| `F1_SCORE` | `"f1_score"` | Harmonic mean of precision and recall |
