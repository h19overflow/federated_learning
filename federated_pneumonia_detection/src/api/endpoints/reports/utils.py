"""Utility functions for report generation."""

import base64
import io
import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def decode_base64_image(base64_string: str) -> Optional[Image.Image]:
    """Decode a base64 string to a PIL Image.

    Args:
        base64_string: Base64 encoded image string

    Returns:
        PIL Image or None if decoding fails
    """
    try:
        # Handle data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_data = base64.b64decode(base64_string)
        image_buffer = io.BytesIO(image_data)
        return Image.open(image_buffer).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None


def prepare_batch_results_for_report(results: list[dict]) -> list[dict]:
    """Convert batch results to the format expected by pdf_report.

    Args:
        results: List of BatchResultItem dicts from the request

    Returns:
        List of dicts in the format expected by generate_batch_summary_report
    """
    formatted_results = []
    for item in results:
        result_dict = {
            "filename": item.get("filename", "Unknown"),
            "success": item.get("success", False),
        }

        if item.get("prediction"):
            pred = item["prediction"]
            result_dict["prediction"] = {
                "predicted_class": pred.get("predicted_class", "N/A"),
                "confidence": pred.get("confidence", 0),
                "pneumonia_probability": pred.get("pneumonia_probability", 0),
                "normal_probability": pred.get("normal_probability", 0),
            }

        if item.get("error"):
            result_dict["error"] = item["error"]

        formatted_results.append(result_dict)

    return formatted_results


def prepare_summary_stats_for_report(summary: dict) -> dict:
    """Convert summary stats to the format expected by pdf_report.

    Args:
        summary: BatchSummaryStats dict from the request

    Returns:
        Dict in the format expected by generate_batch_summary_report
    """
    return {
        "total_images": summary.get("total_images", 0),
        "successful": summary.get("successful", 0),
        "failed": summary.get("failed", 0),
        "pneumonia_count": summary.get("pneumonia_count", 0),
        "normal_count": summary.get("normal_count", 0),
        "avg_confidence": summary.get("avg_confidence", 0),
        "high_risk_count": summary.get("high_risk_count", 0),
    }
