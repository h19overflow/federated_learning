"""
Endpoints for running federated learning training experiments.

This module provides HTTP endpoints to trigger federated machine learning training
on the pneumonia dataset. The training is executed asynchronously in the background,
allowing the endpoint to return immediately while training proceeds.

The federated learning process:
1. Loads dataset metadata and validates paths
2. Partitions data across virtual clients using configured strategy
3. Initializes FederatedTrainer with current configuration settings
4. Runs multiple rounds of federated learning where each client trains locally
5. Aggregates model updates across clients using federated averaging
6. Stores results and global model in configured output directories

Configuration should be set prior to invoking training via the configuration endpoints.
Key federated parameters: num_rounds, num_clients, local_epochs, num_clients
"""

from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form
from typing import Dict, Any
import os
import shutil

import torch

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from federated_pneumonia_detection.src.utils.data_processing import load_metadata
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.control.federated_learning import (
    FederatedTrainer,
)
from .utils import get_websocket_manager
# TODO: update the federated trainer in order to use the new websocket sender in the dl_model/utils/data/websockets_metrics_sender.py
router = APIRouter(
    prefix="/experiments/federated",
    tags=["experiments", "federated"],
)

logger = get_logger(__name__)


def _run_federated_training_task(
    source_path: str,
    experiment_name: str,
    csv_filename: str,
    websocket_manager: Any = None,
) -> Dict[str, Any]:
    """
    Background task to execute federated training.

    Args:
        source_path: Path to training data directory
        experiment_name: Name identifier for this training run
        csv_filename: Name of the CSV metadata file
        websocket_manager: Optional WebSocket manager for real-time progress updates

    Returns:
        Dictionary containing training results
    """
    import asyncio
    task_logger = get_logger(f"{__name__}._task")

    task_logger.info("=" * 80)
    task_logger.info(
        "FEDERATED LEARNING TRAINING - Pneumonia Detection (Background Task)"
    )
    task_logger.info("=" * 80)

    try:
        source_path = Path(source_path)
        image_dir = source_path / "Images"
        metadata_path = source_path / csv_filename

        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        task_logger.info("\nLoading configuration...")
        constants = SystemConstants()
        config = ExperimentConfig()

        task_logger.info(f"  Num clients: {config.num_clients}")
        task_logger.info(f"  Num rounds: {config.num_rounds}")
        task_logger.info(f"  Local epochs: {config.local_epochs}")
        task_logger.info(f"  Learning rate: {config.learning_rate}")
        task_logger.info(f"  Batch size: {config.batch_size}")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            task_logger.info("  Device: GPU (CUDA)")
        else:
            device = torch.device("cpu")
            task_logger.info("  Device: CPU")

        task_logger.info("\nLoading dataset metadata...")
        data_df = load_metadata(metadata_path, constants, task_logger)
        task_logger.info(f"  Total samples: {len(data_df)}")
        class_dist = data_df[constants.TARGET_COLUMN].value_counts().to_dict()
        task_logger.info(f"  Class distribution: {class_dist}")

        task_logger.info("\nInitializing FederatedTrainer...")
        trainer = FederatedTrainer(
            config=config,
            constants=constants,
            device=device,
            websocket_manager=websocket_manager,
        )

        task_logger.info("\nStarting federated learning simulation...")
        task_logger.info("-" * 80)

        results = trainer.train(
            data_df=data_df,
            image_dir=image_dir,
            experiment_name=experiment_name,
        )

        task_logger.info("\n" + "=" * 80)
        task_logger.info("FEDERATED TRAINING COMPLETED SUCCESSFULLY!")
        task_logger.info("=" * 80)

        # Send completion status via WebSocket
        if websocket_manager:
            try:
                completion_message = {
                    "type": "status",
                    "data": {
                        "status": "completed",
                        "message": "Federated training completed successfully",
                        "experiment_name": experiment_name,
                    },
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                }
                asyncio.run(websocket_manager.broadcast(completion_message, experiment_name))
                task_logger.info("Sent completion status via WebSocket")
            except Exception as ws_error:
                task_logger.warning(f"Failed to send WebSocket completion: {ws_error}")

        return results

    except Exception as e:
        task_logger.error("FEDERATED TRAINING FAILED!")
        task_logger.error(f"Error: {type(e).__name__}: {str(e)}")

        import traceback

        task_logger.error("\nFull traceback:")
        task_logger.error(traceback.format_exc())

        # Send error status via WebSocket
        if websocket_manager:
            try:
                error_message = {
                    "type": "status",
                    "data": {
                        "status": "failed",
                        "message": f"Federated training failed: {str(e)}",
                        "experiment_name": experiment_name,
                    },
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                }
                asyncio.run(websocket_manager.broadcast(error_message, experiment_name))
            except Exception as ws_error:
                task_logger.warning(f"Failed to send WebSocket error: {ws_error}")

        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
        }


@router.post("/train")
async def start_federated_training(
    background_tasks: BackgroundTasks,
    data_zip: UploadFile = File(...),
    experiment_name: str = Form("pneumonia_federated"),
    csv_filename: str = Form("stage2_train_metadata.csv"),
) -> Dict[str, Any]:
    """
    Start federated learning training in the background with uploaded data.

    Initiates a federated machine learning training process using the current
    configuration settings. The training runs asynchronously, allowing this endpoint
    to return immediately.

    **Federated Learning Overview:**
    Federated learning trains a shared model across multiple virtual clients without
    centralizing the training data. The process involves:
    - Data is partitioned across virtual clients based on configuration
    - Each round: clients train locally on their data, then send model updates
    - Server aggregates updates using federated averaging (FedAvg algorithm)
    - Global model is updated and distributed back to clients for next round

    **Training Process:**
    - Extracts uploaded data archive (Images/ and metadata CSV)
    - Partitions data into num_clients virtual clients
    - Runs num_rounds of federated learning iterations
    - Each client performs local_epochs training iterations per round
    - Global model aggregation after each round using FedAvg
    - Applies data augmentation and preprocessing based on configuration
    - Saves global model checkpoints and training logs

    **Prerequisites:**
    - Configuration should be set via `/configuration/set_configuration` endpoint
    - Important federated parameters: num_clients, num_rounds, local_epochs, learning_rate
    - Upload a ZIP file containing Images/ directory and metadata CSV

    **Parameters:**
    - `data_zip`: ZIP file containing Images/ directory and metadata CSV (required)
    - `experiment_name`: Identifier for this training run (default: "pneumonia_federated")
    - `csv_filename`: Metadata CSV filename inside archive (default: "stage2_train_metadata.csv")

    **Response:**
    Returns immediately with confirmation that training has been queued. Check logs
    and results for training progress.

    **Configuration Recommendations:**
    - num_clients: 2-10 (number of virtual clients)
    - num_rounds: 5-20 (communication rounds)
    - local_epochs: 1-15 (local training per round)
    - learning_rate: 0.001-0.01 (recommend lower than centralized)

    **Status Tracking:**
    Monitor training progress through:
    - Log files generated during training
    - Checkpoint files saved after each round
    - Global model updates in results directory
    """
    import zipfile
    import tempfile

    temp_dir = None
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

        logger.info(f"Received request to start federated training: {experiment_name}")
        logger.info(f"Extracted data to: {source_path}")

        background_tasks.add_task(
            _run_federated_training_task,
            source_path=source_path,
            experiment_name=experiment_name,
            csv_filename=csv_filename,
            websocket_manager=get_websocket_manager(),
        )

        return {
            "message": "Federated learning training started successfully",
            "experiment_name": experiment_name,
            "status": "queued",
        }
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
