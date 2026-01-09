"""Dataset statistics and information utilities."""

from pathlib import Path
import numpy as np
from PIL import Image
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

logger = get_logger(__name__)


def get_class_distribution(labels: np.ndarray, valid_indices: np.ndarray) -> dict:
    """
    Get class distribution of valid samples.

    Args:
        labels: Array of all labels
        valid_indices: Array of valid sample indices

    Returns:
        Dictionary with class counts
    """
    if len(valid_indices) == 0:
        return {}

    valid_labels = labels[valid_indices]
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    return dict(zip(unique_labels.astype(int), counts))


def get_sample_info(
    idx: int,
    valid_indices: np.ndarray,
    filenames: np.ndarray,
    labels: np.ndarray,
    image_dir: Path,
) -> dict:
    """
    Get detailed information about a sample.

    Args:
        idx: Sample index
        valid_indices: Array of valid sample indices
        filenames: Array of all filenames
        labels: Array of all labels
        image_dir: Directory containing images

    Returns:
        Dictionary with sample information

    Raises:
        IndexError: If index is out of bounds
    """
    if idx >= len(valid_indices) or idx < 0:
        logger.error(f"Index {idx} out of bounds")
        raise IndexError(f"Index {idx} out of bounds")

    actual_idx = valid_indices[idx]
    filename = filenames[actual_idx]
    label = labels[actual_idx]
    image_path = image_dir / filename

    info = {
        "index": idx,
        "actual_index": actual_idx,
        "filename": filename,
        "label": label,
        "image_path": str(image_path),
        "exists": image_path.exists(),
    }

    # Add image info if file exists
    if image_path.exists():
        try:
            with Image.open(image_path) as img:
                info.update(
                    {
                        "image_size": img.size,
                        "image_mode": img.mode,
                        "image_format": img.format,
                    }
                )
        except Exception as e:
            info["image_error"] = str(e)

    return info


def get_memory_usage_estimate(
    valid_indices: np.ndarray,
    filenames: np.ndarray,
    image_dir: Path,
    color_mode: str,
) -> dict:
    """
    Estimate memory usage of the dataset.

    Args:
        valid_indices: Array of valid sample indices
        filenames: Array of all filenames
        image_dir: Directory containing images
        color_mode: Color mode ('RGB' or 'L')

    Returns:
        Dictionary with memory usage estimates
    """
    if len(valid_indices) == 0:
        logger.info("No valid indices in dataset")
        return {"total_samples": 0, "estimated_memory_mb": 0}

    # Sample a few images to estimate average size
    sample_size = min(10, len(valid_indices))
    total_pixels = 0

    for i in range(sample_size):
        try:
            actual_idx = valid_indices[i]
            filename = filenames[actual_idx]
            image_path = image_dir / filename

            with Image.open(image_path) as img:
                total_pixels += img.size[0] * img.size[1]

        except Exception:
            logger.error(f"Error validating image {image_path}")
            continue

    if sample_size > 0:
        avg_pixels = total_pixels / sample_size
        channels = 3 if color_mode == "RGB" else 1
        bytes_per_pixel = 4  # Assuming float32

        estimated_mb_per_image = (avg_pixels * channels * bytes_per_pixel) / (
            1024 * 1024
        )
        total_estimated_mb = estimated_mb_per_image * len(valid_indices)

    return {
        "total_samples": len(valid_indices),
        "avg_pixels_per_image": avg_pixels if sample_size > 0 else 0,
        "estimated_mb_per_image": estimated_mb_per_image,
        "estimated_total_memory_mb": total_estimated_mb,
        "color_mode": color_mode,
    }
