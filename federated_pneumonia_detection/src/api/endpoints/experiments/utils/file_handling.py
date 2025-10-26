"""
File handling utilities for experiment endpoints.

Provides functions for handling uploaded ZIP files and extracting datasets.
"""

import os
import shutil
import tempfile
import zipfile
from typing import Optional

from fastapi import UploadFile


async def prepare_zip(data_zip: UploadFile, logger, experiment_name: str) -> str:
    """
    Extract uploaded ZIP file and return the extraction path.

    Handles temporary directory creation, ZIP extraction, and error cleanup.

    Args:
        data_zip: Uploaded ZIP file containing Images/ and metadata CSV
        logger: Logger instance for logging operations
        experiment_name: Name of the experiment for logging purposes

    Returns:
        Path to extracted directory containing Images/ and metadata files

    Raises:
        Exception: If ZIP extraction or file operations fail
    """
    temp_dir: Optional[str] = None
    try:
        # Create temp directory for extraction
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, data_zip.filename)

        # Save uploaded file
        with open(zip_path, "wb") as f:
            content = await data_zip.read()
            f.write(content)

        # Extract archive
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        source_path = extract_path

        logger.info(f"Received request to start training: {experiment_name}")
        logger.info(f"Extracted data to: {source_path}")
        return source_path
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
