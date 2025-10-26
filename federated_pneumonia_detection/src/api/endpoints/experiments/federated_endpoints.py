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

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form
from typing import Dict, Any

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
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
    try:
        source_path = await prepare_zip(data_zip, logger, experiment_name)
        background_tasks.add_task(
            run_federated_training_task,
            source_path=source_path,
            experiment_name=experiment_name,
            csv_filename=csv_filename,
        )

        return {
            "message": "Federated learning training started successfully",
            "experiment_name": experiment_name,
            "status": "queued",
        }
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise
