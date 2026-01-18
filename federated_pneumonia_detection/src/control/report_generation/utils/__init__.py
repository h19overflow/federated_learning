"""PDF Report Generation Utilities.

Provides modular components for building clinical PDF reports.
"""

from federated_pneumonia_detection.src.control.report_generation.utils.constants import (
    BRAND_COLOR,
    DANGER_COLOR,
    LIGHT_BG,
    SUCCESS_COLOR,
    WARNING_COLOR,
)
from federated_pneumonia_detection.src.control.report_generation.utils.images import (
    base64_to_image,
    pil_to_reportlab_image,
)
from federated_pneumonia_detection.src.control.report_generation.utils.styles import (
    get_styles,
)

__all__ = [
    "BRAND_COLOR",
    "DANGER_COLOR",
    "LIGHT_BG",
    "SUCCESS_COLOR",
    "WARNING_COLOR",
    "base64_to_image",
    "pil_to_reportlab_image",
    "get_styles",
]
