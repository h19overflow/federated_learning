"""
Dataset Sampling Script

Purpose: Sample 5% of training/testing data with corresponding images and masks
for faster development testing while maintaining dataset structure.

Dependencies: pandas, shutil, pathlib
Role: Creates a smaller subset of the dataset for testing purposes
"""

import pandas as pd
import shutil
from pathlib import Path
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSampler:
    """
    Samples a percentage of dataset rows and copies corresponding files.

    Maintains data integrity by preserving all rows for sampled patient IDs.
    """

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        sample_percentage: float = 0.05
    ):
        """
        Initialize the dataset sampler.

        Args:
            source_dir: Path to original Training directory
            output_dir: Path where sampled data will be saved
            sample_percentage: Percentage of data to sample (0.0 to 1.0)

        Raises:
            ValueError: If sample_percentage is not between 0 and 1
        """
        if not 0 < sample_percentage <= 1:
            raise ValueError("sample_percentage must be between 0 and 1")

        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.sample_percentage = sample_percentage

        # Validate source directory exists
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        # Create output directory structure
        self._create_output_structure()

    def _create_output_structure(self) -> None:
        """Create output directory structure matching source."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "Images").mkdir(exist_ok=True)
        (self.output_dir / "Masks").mkdir(exist_ok=True)
        logger.info(f"Created output directory structure at {self.output_dir}")

    def _sample_patient_ids(self, df: pd.DataFrame) -> set:
        """
        Sample unique patient IDs from dataframe.

        Args:
            df: DataFrame containing patientId column

        Returns:
            Set of sampled patient IDs
        """
        unique_patients = df['patientId'].unique()
        sample_size = max(1, int(len(unique_patients) * self.sample_percentage))
        sampled_patients = pd.Series(unique_patients).sample(
            n=sample_size,
            random_state=42  # For reproducibility
        )
        return set(sampled_patients)

    def _copy_image_file(
        self,
        patient_id: str,
        subfolder: str
    ) -> bool:
        """
        Copy image or mask file to output directory.

        Args:
            patient_id: Patient identifier
            subfolder: "Images" or "Masks"

        Returns:
            True if file was copied, False if file doesn't exist
        """
        source_file = self.source_dir / subfolder / f"{patient_id}.png"
        dest_file = self.output_dir / subfolder / f"{patient_id}.png"

        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            return True
        else:
            logger.warning(f"File not found: {source_file}")
            return False

    def process_csv_file(
        self,
        csv_filename: str
    ) -> Tuple[pd.DataFrame, int, int]:
        """
        Process a single CSV file and sample its data.

        Args:
            csv_filename: Name of CSV file (e.g., "stage2_train_metadata.csv")

        Returns:
            Tuple of (sampled_df, original_row_count, sampled_row_count)
        """
        # Read CSV
        source_csv = self.source_dir / csv_filename
        df = pd.read_csv(source_csv)
        original_count = len(df)

        logger.info(f"Processing {csv_filename}: {original_count} rows")

        # Sample patient IDs
        sampled_ids = self._sample_patient_ids(df)
        logger.info(f"Sampled {len(sampled_ids)} unique patient IDs")

        # Filter dataframe to only sampled patient IDs
        sampled_df = df[df['patientId'].isin(sampled_ids)].copy()
        sampled_count = len(sampled_df)

        # Save sampled CSV
        output_csv = self.output_dir / csv_filename
        sampled_df.to_csv(output_csv, index=False)
        logger.info(f"Saved {sampled_count} rows to {output_csv}")

        return sampled_df, original_count, sampled_count

    def copy_images_for_patients(self, patient_ids: set) -> dict:
        """
        Copy images and masks for given patient IDs.

        Args:
            patient_ids: Set of patient IDs to copy files for

        Returns:
            Dictionary with copy statistics
        """
        stats = {
            'images_copied': 0,
            'masks_copied': 0,
            'images_missing': 0,
            'masks_missing': 0
        }

        for patient_id in patient_ids:
            # Copy image
            if self._copy_image_file(patient_id, "Images"):
                stats['images_copied'] += 1
            else:
                stats['images_missing'] += 1

            # Copy mask
            if self._copy_image_file(patient_id, "Masks"):
                stats['masks_copied'] += 1
            else:
                stats['masks_missing'] += 1

        return stats

    def sample_dataset(self) -> dict:
        """
        Execute full dataset sampling process.

        Returns:
            Dictionary containing sampling statistics
        """
        logger.info(f"Starting dataset sampling ({self.sample_percentage*100}%)")

        all_stats = {}
        all_patient_ids = set()

        # Process each CSV file
        csv_files = [
            "stage2_train_metadata.csv",
            "stage2_test_metadata.csv"
        ]

        for csv_file in csv_files:
            csv_path = self.source_dir / csv_file

            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}, skipping")
                continue

            sampled_df, original_count, sampled_count = self.process_csv_file(csv_file)

            # Collect patient IDs
            patient_ids = set(sampled_df['patientId'].unique())
            all_patient_ids.update(patient_ids)

            all_stats[csv_file] = {
                'original_rows': original_count,
                'sampled_rows': sampled_count,
                'unique_patients': len(patient_ids),
                'percentage': (sampled_count / original_count * 100) if original_count > 0 else 0
            }

        # Copy images and masks
        logger.info(f"Copying files for {len(all_patient_ids)} unique patients...")
        file_stats = self.copy_images_for_patients(all_patient_ids)
        all_stats['files'] = file_stats

        return all_stats


def main():
    """Main execution function."""
    # Configuration
    SOURCE_DIR = Path("C:/Users/User/Projects/FYP2/Training")
    OUTPUT_DIR = Path("C:/Users/User/Projects/FYP2/Training_Sample_5pct")
    SAMPLE_PERCENTAGE = 0.05

    try:
        # Create sampler and execute
        sampler = DatasetSampler(SOURCE_DIR, OUTPUT_DIR, SAMPLE_PERCENTAGE)
        stats = sampler.sample_dataset()

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("SAMPLING COMPLETE - Summary:")
        logger.info("="*60)

        for csv_file, csv_stats in stats.items():
            if csv_file == 'files':
                continue
            logger.info(f"\n{csv_file}:")
            logger.info(f"  Original rows: {csv_stats['original_rows']}")
            logger.info(f"  Sampled rows: {csv_stats['sampled_rows']}")
            logger.info(f"  Unique patients: {csv_stats['unique_patients']}")
            logger.info(f"  Percentage: {csv_stats['percentage']:.2f}%")

        if 'files' in stats:
            logger.info(f"\nFiles:")
            logger.info(f"  Images copied: {stats['files']['images_copied']}")
            logger.info(f"  Masks copied: {stats['files']['masks_copied']}")
            logger.info(f"  Images missing: {stats['files']['images_missing']}")
            logger.info(f"  Masks missing: {stats['files']['masks_missing']}")

        logger.info(f"\nOutput directory: {OUTPUT_DIR}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Error during sampling: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
