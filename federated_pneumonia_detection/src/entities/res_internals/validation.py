"""Parameter validation for ResNetWithCustomHead."""

from typing import Any


def validate_parameters(
    num_classes: int,
    dropout_rate: float,
    fine_tune_layers_count: Any,
    logger: Any,
) -> None:
    """
    Validate initialization parameters.

    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate value
        fine_tune_layers_count: Fine-tuning layers count
        logger: Logger instance

    Raises:
        ValueError: If any parameter is invalid
    """
    if num_classes <= 0:
        logger.error("num_classes must be positive")
        raise ValueError("num_classes must be positive")

    if not 0.0 <= dropout_rate <= 1.0:
        logger.error("dropout_rate must be between 0.0 and 1.0")
        raise ValueError("dropout_rate must be between 0.0 and 1.0")

    if not isinstance(fine_tune_layers_count, int):
        logger.error("fine_tune_layers_count must be an integer")
        raise ValueError("fine_tune_layers_count must be an integer")
