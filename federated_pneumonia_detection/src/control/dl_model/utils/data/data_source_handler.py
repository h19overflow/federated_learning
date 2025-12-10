"""
Handles data source extraction and validation for training datasets from zip files or directories.

This unified handler provides a robust mechanism to extract and validate training datasets
from different source types with comprehensive logging and error handling.
"""

import os
import zipfile
import tempfile
import shutil
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger


class DataSourceExtractor:
    """
    Extracts and validates training datasets from zip files or directories.

    Handles both zip archives and directory-based data sources with a unified interface.
    Provides methods for extraction, validation, and cleanup.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize DataSourceExtractor.

        Args:
            logger: Optional logger instance. If not provided, uses default logger.
        """
        self.logger = logger or get_logger(__name__)
        self.temp_extract_dir: Optional[str] = None
        self._image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def extract_and_validate(
        self,
        source_path: str,
        csv_filename: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Extract and validate data source contents.

        Auto-detects whether source is a zip file or directory.
        Extracts zip to temporary directory if necessary.

        Args:
            source_path: Path to data source (zip file or directory)
            csv_filename: Optional specific CSV filename to find

        Returns:
            Tuple of (image_directory, csv_filepath)

        Raises:
            FileNotFoundError: If source path does not exist
            ValueError: If required files are missing
        """
        self.logger.info(f"Validating data source: {source_path}")
        # Detect source type
        is_zip = source_path.lower().endswith('.zip')
        source_type = "ZIP file" if is_zip else "Directory"
        self.logger.info(f"Detected source type: {source_type}")

        try:
            if is_zip:
                self.logger.info(f"Processing ZIP file...")
                result = self._process_zip(source_path, csv_filename)
            else:
                self.logger.info(f"Processing directory...")
                result = self._process_directory(source_path, csv_filename)

            self.logger.info(f"✓ Data source processing completed successfully")
            return result

        except FileNotFoundError as e:
            self.logger.error(f"✗ File not found during processing: {str(e)}")
            raise
        except ValueError as e:
            self.logger.error(f"✗ Validation error during processing: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"✗ Unexpected error during processing: {type(e).__name__}: {str(e)}")
            raise

    def validate_contents(self, source_path: str) -> Dict[str, Any]:
        """
        Validate source contents without processing.

        Args:
            source_path: Path to data source (zip or directory)

        Returns:
            Dictionary with validation results
        """
        if not os.path.exists(source_path):
            return {'valid': False, 'error': 'Source not found'}

        is_zip = source_path.lower().endswith('.zip')

        try:
            if is_zip:
                return self._validate_zip(source_path)
            else:
                return self._validate_directory(source_path)
        except Exception as e:
            return {'valid': False, 'error': f'Validation failed: {e}'}

    def cleanup(self):
        """
        Clean up temporary extraction directory if it exists.

        Safe to call multiple times. Handles cases where directory might have been deleted.
        """
        if self.temp_extract_dir and os.path.exists(self.temp_extract_dir):
            try:
                shutil.rmtree(self.temp_extract_dir)
                self.logger.info("✓ Temporary directory cleaned up successfully")
                self.temp_extract_dir = None
            except Exception as e:
                self.logger.warning(f"⚠ Failed to cleanup temp directory: {e}")
        elif self.temp_extract_dir:
            self.logger.debug("Temp directory already cleaned or doesn't exist")

    def _process_zip(self, zip_path: str, csv_filename: Optional[str] = None) -> Tuple[str, str]:
        """Process and extract contents from a zip file."""
        try:
            self.temp_extract_dir = tempfile.mkdtemp(prefix="pneumonia_training_")
            self.logger.info(f"  Created temporary directory: {self.temp_extract_dir}")

            self.logger.info(f"  Extracting ZIP contents...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_count = len(zip_ref.namelist())
                self.logger.info(f"  ZIP contains {file_count} files")
                zip_ref.extractall(self.temp_extract_dir)
            self.logger.info(f"  ✓ Extraction completed")

            self.logger.info(f"  Searching for CSV file...")
            csv_path = self._find_csv_file(self.temp_extract_dir, csv_filename)
            self.logger.info(f"  ✓ CSV file found: {os.path.basename(csv_path)}")

            self.logger.info(f"  Searching for image directory...")
            image_dir = self._find_image_directory(self.temp_extract_dir)
            self.logger.info(f"  ✓ Image directory found")

            return image_dir, csv_path
        except Exception as e:
            self.logger.error(f"  ✗ ZIP processing failed: {type(e).__name__}: {str(e)}")
            raise

    def _process_directory(self, directory_path: str, csv_filename: Optional[str] = None) -> Tuple[str, str]:
        """Process contents from a directory."""
        try:
            if not os.path.isdir(directory_path):
                self.logger.error(f"  ✗ Path is not a directory: {directory_path}")
                raise ValueError(f"Path is not a directory: {directory_path}")
            self.logger.info(f"  ✓ Directory verified")

            csv_path = self._find_csv_file(directory_path, csv_filename)
            self.logger.info(f"  ✓ CSV file found: {os.path.basename(csv_path)}")

            image_dir = self._find_image_directory(directory_path)

            return image_dir, csv_path
        except Exception as e:
            self.logger.error(f"  ✗ Directory processing failed: {type(e).__name__}: {str(e)}")
            raise

    def _validate_zip(self, zip_path: str) -> Dict[str, Any]:
        """Validate zip file contents."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

        csv_files = [f for f in file_list if f.endswith('.csv')]
        image_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in self._image_extensions)]

        validation = {
            'valid': len(csv_files) > 0 and len(image_files) > 0,
            'csv_files': csv_files,
            'image_count': len(image_files),
            'total_files': len(file_list),
            'error': None
        }

        if not validation['valid']:
            validation['error'] = f"Missing required files. CSV: {len(csv_files)}, Images: {len(image_files)}"

        return validation

    def _validate_directory(self, directory_path: str) -> Dict[str, Any]:
        """Validate directory contents."""
        if not os.path.isdir(directory_path):
            return {'valid': False, 'error': 'Path is not a directory'}

        csv_files = list(Path(directory_path).rglob("*.csv"))
        image_files = []
        for ext in self._image_extensions:
            image_files.extend(list(Path(directory_path).rglob(f"*{ext}")))

        validation = {
            'valid': len(csv_files) > 0 and len(image_files) > 0,
            'csv_files': [str(f) for f in csv_files],
            'image_count': len(image_files),
            'error': None
        }

        if not validation['valid']:
            validation['error'] = f"Missing required files. CSV: {len(csv_files)}, Images: {len(image_files)}"

        return validation

    def _find_csv_file(self, base_path: str, csv_filename: Optional[str] = None) -> str:
        """Find CSV file in given path."""
        csv_files = list(Path(base_path).rglob("*.csv"))

        if not csv_files:
            self.logger.error(f"    ✗ No CSV files found in {base_path}")
            raise ValueError("No CSV files found")

        if csv_filename:
            matching_files = [f for f in csv_files if f.name == csv_filename]
            if not matching_files:
                available = [f.name for f in csv_files]
                self.logger.error(f"    ✗ Specified CSV not found: {csv_filename}")
            self.logger.debug(f"    ✓ Matched CSV file: {matching_files[0].name}")
            return str(matching_files[0])

        selected = csv_files[0]
        self.logger.debug(f"    Auto-selected first CSV: {selected.name}")
        return str(selected)

    def _find_image_directory(self, base_path: str) -> str:
        """Find image directory in given path."""
        self.logger.debug(f"    Searching for image files in: {base_path}")
        self.logger.debug(f"    Supported extensions: {self._image_extensions}")

        image_files = []
        for ext in self._image_extensions:
            found = list(Path(base_path).rglob(f"*{ext}"))
            if found:
                self.logger.debug(f"    Found {len(found)} {ext} files")
            image_files.extend(found)

        if not image_files:
            self.logger.error(f"    ✗ No image files found in {base_path}")
            self.logger.error(f"    Searched extensions: {self._image_extensions}")
            raise ValueError("No image files found")

        image_dir = str(image_files[0].parent)
        self.logger.info(f"    Found image directory: {image_dir}")
        self.logger.info(f"    Total images found: {len(image_files)}")
        return image_dir
