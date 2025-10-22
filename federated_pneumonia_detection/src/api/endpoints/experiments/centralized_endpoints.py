"""
Endpoints for running centralized training experiments.

This module provides HTTP endpoints to trigger centralized machine learning training
on the pneumonia dataset. The training is executed asynchronously in the background,
allowing the endpoint to return immediately while training proceeds.

The training process:
1. Loads the dataset from the configured Training directory
2. Initializes a CentralizedTrainer with current configuration settings
3. Trains the model using standard supervised learning approach
4. Stores results and checkpoints in configured output directories

Configuration should be set prior to invoking training via the configuration endpoints.
"""

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form
from typing import Dict, Any
import os
import shutil

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from .utils import get_websocket_manager,_run_centralized_training_task, prepare_zip

router = APIRouter(
    prefix="/experiments/centralized",
    tags=["experiments", "centralized"],
)

logger = get_logger(__name__)




@router.post("/train")
async def start_centralized_training(
    background_tasks: BackgroundTasks,
    data_zip: UploadFile = File(...),
    checkpoint_dir: str = Form("results/centralized/checkpoints"),
    logs_dir: str = Form("results/centralized/logs"),
    experiment_name: str = Form("pneumonia_centralized"),
    csv_filename: str = Form("stage2_train_metadata.csv"),
) -> Dict[str, Any]:
    """
    Start centralized training in the background with uploaded data.
    Initiates a centralized machine learning training process using the current
    configuration settings. The training runs asynchronously, allowing this endpoint
    to return immediately.
    **Training Process:**
    - Extracts uploaded data archive (Images/ and metadata CSV)
    - Uses all training samples in a single centralized trainer
    - Applies data augmentation and preprocessing based on configuration
    - Trains using standard supervised learning approach
    - Performs validation during training with early stopping
    - Saves best model checkpoints and training logs

    **Parameters:**
    - `data_zip`: ZIP file containing Images/ directory and metadata CSV (required)
    - `checkpoint_dir`: Where to save model checkpoints (default: "results/centralized/checkpoints")
    - `logs_dir`: Where to save training logs (default: "results/centralized/logs")
    - `experiment_name`: Identifier for this training run (default: "pneumonia_centralized")
    - `csv_filename`: Metadata CSV filename inside archive (default: "stage2_train_metadata.csv")
    **Status Tracking:**
    Monitor training progress through:
    - Log files at `{logs_dir}/`
    - Checkpoint files at `{checkpoint_dir}/`
    """
    temp_dir = None
    try:
        # Create temp directory for extraction
        source_path = await prepare_zip(data_zip,logger,experiment_name)
        background_tasks.add_task(
            _run_centralized_training_task,
            source_path=source_path,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir,
            experiment_name=experiment_name,
            csv_filename=csv_filename,
        )

        return {
            "message": "Centralized training started successfully",
            "experiment_name": experiment_name,
            "checkpoint_dir": checkpoint_dir,
            "logs_dir": logs_dir,
            "status": "queued",
        }
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
