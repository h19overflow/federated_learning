"""
Dataset profiler for EDA and research paper documentation.

Generates comprehensive dataset statistics for Section 5.2 of research papers,
including data sources, cleaning steps, and feature engineering documentation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from analysis.preprocessing.data_loader import AnalysisDataLoader


class DataProfiler:
    """
    Generates comprehensive dataset profiles for research documentation.

    Produces statistics and documentation for:
    - Data sources and collection
    - Data cleaning steps
    - Feature engineering transformations
    - Class distribution analysis
    """

    def __init__(
        self,
        data_loader: AnalysisDataLoader,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize profiler with data loader.

        Args:
            data_loader: AnalysisDataLoader instance with loaded data
            logger: Optional logger instance
        """
        self.data_loader = data_loader
        self.logger = logger or logging.getLogger(__name__)
        self._profile: Optional[Dict[str, Any]] = None

    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate comprehensive dataset profile.

        Returns:
            Dictionary containing all profile information
        """
        df = self.data_loader.dataframe
        if df is None:
            df = self.data_loader.load_full_dataset()

        stats = self.data_loader.get_statistics()
        target_col = self.data_loader.config.get("columns.target")

        self._profile = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source_path": str(self.data_loader.source_path),
                "csv_filename": self.data_loader.csv_filename,
            },
            "dataset_overview": {
                "total_samples": stats["total_samples"],
                "num_features": len(stats["columns"]),
                "columns": stats["columns"],
            },
            "class_distribution": self._analyze_class_distribution(df, target_col, stats),
            "data_quality": self._analyze_data_quality(df, stats),
            "preprocessing_steps": self._document_preprocessing_steps(),
            "feature_engineering": self._document_feature_engineering(),
        }

        self.logger.info(f"Generated profile for {stats['total_samples']} samples")
        return self._profile

    def _analyze_class_distribution(
        self, df: pd.DataFrame, target_col: str, stats: Dict
    ) -> Dict:
        """Analyze class distribution and imbalance."""
        class_dist = stats["class_distribution"]
        total = stats["total_samples"]

        class_analysis = {}
        for label, count in class_dist.items():
            class_analysis[str(label)] = {
                "count": count,
                "percentage": round(count / total * 100, 2),
            }

        return {
            "classes": class_analysis,
            "num_classes": len(class_dist),
            "balance_ratio": round(stats.get("class_balance_ratio", 1.0), 4),
            "is_imbalanced": stats.get("class_balance_ratio", 1.0) < 0.5,
            "majority_class": max(class_dist, key=class_dist.get),
            "minority_class": min(class_dist, key=class_dist.get),
        }

    def _analyze_data_quality(self, df: pd.DataFrame, stats: Dict) -> Dict:
        """Analyze data quality metrics."""
        missing = stats["missing_values"]
        total_missing = sum(missing.values())

        return {
            "missing_values": missing,
            "total_missing": total_missing,
            "missing_percentage": round(total_missing / (len(df) * len(df.columns)) * 100, 4),
            "complete_cases": len(df.dropna()),
            "complete_cases_percentage": round(len(df.dropna()) / len(df) * 100, 2),
            "duplicates": int(df.duplicated().sum()),
        }

    def _document_preprocessing_steps(self) -> Dict:
        """Document preprocessing steps applied to the data."""
        return {
            "steps": [
                {
                    "step": 1,
                    "name": "Load CSV Metadata",
                    "description": "Load patient metadata from CSV file",
                    "function": "load_metadata()",
                },
                {
                    "step": 2,
                    "name": "Generate Filenames",
                    "description": "Create image filenames from patient IDs with .jpg extension",
                    "transformation": "patient_id + '.jpg' -> filename",
                },
                {
                    "step": 3,
                    "name": "Target Encoding",
                    "description": "Convert target column to string labels",
                    "values": {"0": "Normal", "1": "Pneumonia"},
                },
                {
                    "step": 4,
                    "name": "Validation",
                    "description": "Check for missing values in critical columns",
                    "columns_checked": ["patient_id", "Target", "filename"],
                },
            ],
            "data_cleaning": {
                "missing_value_handling": "Rows with missing critical values are rejected",
                "duplicate_handling": "Duplicates preserved (same patient may have multiple views)",
                "outlier_handling": "No outlier removal applied to metadata",
            },
        }

    def _document_feature_engineering(self) -> Dict:
        """Document feature engineering transformations."""
        return {
            "image_preprocessing": {
                "resize": "224x224 pixels",
                "normalization": "ImageNet mean/std normalization",
                "color_mode": "RGB (3 channels)",
            },
            "augmentation_training": {
                "random_resized_crop": {"scale": [0.8, 1.0]},
                "random_horizontal_flip": {"probability": 0.5},
                "random_rotation": {"degrees": 15},
                "color_jitter": {
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.1,
                    "hue": 0.1,
                },
            },
            "augmentation_validation": {
                "resize": "256x256 pixels",
                "center_crop": "224x224 pixels",
            },
        }

    def save_profile(self, output_path: Path) -> None:
        """
        Save profile to JSON file.

        Args:
            output_path: Path to save JSON file
        """
        if self._profile is None:
            self.generate_profile()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self._profile, f, indent=2, default=str)

        self.logger.info(f"Profile saved to {output_path}")

    @property
    def profile(self) -> Optional[Dict[str, Any]]:
        """Get generated profile."""
        return self._profile
