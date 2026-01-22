"""Endpoints for running federated training experiments."""

from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from .utils import prepare_zip, run_federated_training_task

router = APIRouter(
    prefix="/experiments/federated",
    tags=["experiments", "federated"],
)

logger = get_logger(__name__)


@router.post("/train")
async def start_federated_training(
    background_tasks: BackgroundTasks,
    data_zip: UploadFile = File(...),
    experiment_name: str = Form("pneumonia_federated"),
    csv_filename: str = Form("stage2_train_metadata.csv"),
    num_server_rounds: int = Form(3),
) -> Dict[str, Any]:
    """
    Start federated training in the background with uploaded data.

    Initiates a federated machine learning training process using the Flower framework.
    The training runs asynchronously, allowing this endpoint to return immediately.

    **Training Process:**
    - Extracts uploaded data archive (Images/ and metadata CSV)
    - Launches federated learning clients on configured compute resources
    - Applies data partitioning and preprocessing based on configuration
    - Trains using federated averaging strategy across multiple clients
    - Performs validation during training with early stopping
    - Saves best model checkpoints and training logs

    **Parameters:**
    - `data_zip`: ZIP file containing Images/ directory and metadata CSV (required)
    - `experiment_name`: Identifier for this training run (default: "pneumonia_federated")
    - `csv_filename`: Metadata CSV filename inside archive (default: "stage2_train_metadata.csv")
    - `num_server_rounds`: Number of federated learning rounds (default: 3)

    **Status Tracking:**
    Monitor training progress through:
    - Log files in configured logs directory
    - Checkpoint files in configured checkpoint directory
    - Process status via system utilities

    **Returns:**
    - `message`: Success message indicating training has been queued
    - `experiment_name`: The identifier used for this training run
    - `num_server_rounds`: Number of rounds that will be executed
    - `status`: "queued" indicating the task has been accepted
    """
     _temp_dir = None
     try:
         source_path = await prepare_zip(data_zip, logger, experiment_name)

         background_tasks.add_task(
             run_federated_training_task,
             source_path=source_path,
             experiment_name=experiment_name,
             csv_filename=csv_filename,
             num_server_rounds=num_server_rounds,
         )

        logger.info(
            f"Federated training queued: {experiment_name} (rounds={num_server_rounds})",
        )

        return {
            "message": "Federated training started successfully",
            "experiment_name": experiment_name,
            "num_server_rounds": num_server_rounds,
            "status": "queued",
        }

    except Exception as e:
        logger.error(f"Error starting federated training: {str(e)}", exc_info=True)
        raise
