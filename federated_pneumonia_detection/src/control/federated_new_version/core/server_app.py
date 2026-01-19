from flwr.app import ArrayRecord, Context
from flwr.serverapp import ServerApp, Grid
from datetime import datetime
import os
from pathlib import Path
import torch
from federated_pneumonia_detection.src.control.federated_new_version.core.custom_strategy import (
    ConfigurableFedAvg,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
)
from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
    read_configs_to_toml,
)
from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.internals.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud

from federated_pneumonia_detection.src.control.federated_new_version.core.server_evaluation import (
    create_central_evaluate_fn,
)
from federated_pneumonia_detection.src.internals.loggers.logger import setup_logger
from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
    _convert_metric_record_to_dict,
    _persist_server_evaluations,
)

# Setup logger for server app
logger = setup_logger(__name__)

app = ServerApp()


@app.lifespan()
def lifespan(app: ServerApp):
    """Lifecycle management for ServerApp.

    Note: Configuration sync should happen BEFORE Flower starts (in federated_tasks.py).
    This lifespan hook runs after Flower has already loaded pyproject.toml,
    so we just verify and log the configuration here.
    """
    logger.info("=" * 80)
    logger.info("SERVER LIFESPAN: Starting up...")
    logger.info("=" * 80)

    # Verify configuration is in sync
    logger.info("Verifying configuration synchronization...")
    flwr_configs = read_configs_to_toml()

    if flwr_configs:
        logger.info(f"[OK] Configuration verified: {flwr_configs}")
        logger.info(
            "NOTE: Config should have been synced to pyproject.toml before Flower started"
        )
    else:
        logger.warning("[WARN] No federated configs found in default_config.yaml")

    yield

    logger.info("=" * 80)
    logger.info("SERVER LIFESPAN: Shutting down...")
    logger.info("=" * 80)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp.

    Following Flower conventions:
    - Initialize global model and convert to ArrayRecord
    - Configure FedAvg strategy with proper parameters
    - Use strategy.start() to run federated learning
    - Provide optional evaluate_fn for server-side evaluation
    """

    num_rounds: int = context.run_config["num-server-rounds"]  # Read run config
    num_clients: int = len(list(grid.get_node_ids()))

    logger.info("=" * 80)
    logger.info("FEDERATED LEARNING SESSION STARTING")
    logger.info("=" * 80)
    logger.info(f"Configuration: {num_clients} clients, {num_rounds} rounds")
    logger.info(f"Context run_config: {context.run_config}")

    # Create the run in database BEFORE training starts
    logger.info("Creating federated training run in database...")
    db = get_session()
    try:
        run_data = {
            "training_mode": "federated",
            "status": "in_progress",
            "start_time": datetime.now(),
            "wandb_id": f"federated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "source_path": "federated_training",
        }
        logger.debug(f"Run data to create: {run_data}")
        new_run = run_crud.create(db, **run_data)
        db.commit()
        run_id = new_run.id
        logger.info(f"[OK] Successfully created run with id={run_id}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to create run in database: {e}", exc_info=True)
        db.rollback()
        run_id = None
    finally:
        db.close()

    # Load ConfigManager FIRST (before any config access)
    # Make config path configurable via environment or use default relative path
    default_config_path = (
        Path(__file__).parent.parent.parent.parent.parent / "config" / "default_config.yaml"
    )
    config_path = os.getenv("CONFIG_PATH", str(default_config_path))
    config_manager = ConfigManager(config_path=str(config_path))

    # Override config from environment variables (for analysis reproducibility)
    # This allows the analysis module to control seeds and parameters via subprocess
    fl_seed = os.getenv("FL_SEED")
    if fl_seed:
        seed_value = int(fl_seed)
        config_manager.set("experiment.seed", seed_value)
        config_manager.set("system.seed", seed_value)
        logger.info(f"[ENV OVERRIDE] Using seed from FL_SEED: {seed_value}")

    # Get seed from config (may have been overridden by FL_SEED env var)
    experiment_seed = config_manager.get("experiment.seed", 42)

    # Create training configuration to send to clients
    train_config = {
        "file_path": config_manager.get("experiment.file-path"),
        "image_dir": config_manager.get("experiment.image-dir"),
        "num_partitions": num_clients,
        "run_id": run_id,  # Pass run_id to clients so they don't create their own
        "seed": experiment_seed,  # Pass seed for reproducible data splits
    }

    # Create evaluation configuration to send to clients
    eval_config = {
        "csv_path": config_manager.get("experiment.file-path"),
        "image_dir": config_manager.get("experiment.image-dir"),
    }

    # Use FL_RUN_ID for result file naming if provided (analysis compatibility)
    analysis_run_id = os.getenv("FL_RUN_ID")
    if analysis_run_id:
        logger.info(f"[ENV OVERRIDE] Using run_id from FL_RUN_ID: {analysis_run_id}")

    # Initialize enhanced model with balanced configuration (matches centralized_trainer.py)
    # Use uniform class weights for server initialization (clients use their own local weights)
    # Class weights don't affect server-side evaluation metrics, only loss computation
    num_classes = 2  # Binary classification: Normal vs Pneumonia
    class_weights = torch.ones(num_classes)

    global_model = LitResNetEnhanced(
        config=config_manager,
        class_weights_tensor=class_weights,
        use_focal_loss=True,
        focal_alpha=0.6,  # Balanced between recall and precision
        focal_gamma=1.5,  # Moderate focus on hard examples
        label_smoothing=0.05,  # Mild regularization
        use_cosine_scheduler=True,
        monitor_metric="val_recall",  # Consistent with centralized training
    )
    logger.info("Initialized LitResNetEnhanced with balanced focal loss configuration")

    arrays = ArrayRecord(global_model.state_dict())

    # Initialize WebSocket sender to broadcast training mode
    logger.info("Initializing WebSocket sender for real-time metrics...")
    ws_sender = MetricsWebSocketSender("ws://localhost:8765")

    # Signal to frontend that this is federated training
    logger.info(
        f"Broadcasting training mode: {num_clients} clients, {num_rounds} rounds"
    )
    ws_sender.send_training_mode(
        is_federated=True, num_rounds=num_rounds, num_clients=num_clients
    )

    # Create centralized evaluation function for server-side evaluation
    # This evaluates the global model on a held-out test set after each round
    # IMPORTANT: Must be created BEFORE strategy initialization
    logger.info("Creating server-side evaluation function...")
    central_evaluate_fn = create_central_evaluate_fn(
        config_manager=config_manager,
        csv_path=config_manager.get("experiment.file-path"),
        image_dir=config_manager.get("experiment.image-dir"),
    )
    logger.info("[OK] Server evaluation function created")

    # Initialize ConfigurableFedAvg strategy with configs
    # Following Flower conventions:
    # - fraction_train: fraction of available clients to use for training each round
    # - fraction_evaluate: fraction of available clients to use for evaluation each round
    # - train_config/eval_config: passed to clients via Message.content["config"]
    # - FedAvg uses 'num_examples' key from client metrics for weighted aggregation
    # - evaluate_fn: server-side evaluation function (NEW in Flower 1.0+)
    logger.info("Initializing FedAvg strategy...")
    strategy = ConfigurableFedAvg(
        fraction_train=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        train_config=train_config,
        eval_config=eval_config,
        websocket_uri="ws://localhost:8765",
        run_id=run_id,  # Pass run_id for database persistence
    )

    # Set total rounds for progress tracking in strategy
    strategy.set_total_rounds(num_rounds)
    logger.info(
        "Strategy configured: FedAvg with weighted aggregation by num_examples + server evaluation"
    )

    # Start strategy, run FedAvg for `num_rounds`
    # Following Flower conventions:
    # 1. strategy.start() orchestrates the entire federated learning process
    # 2. Each round: configure_train -> clients train -> aggregate_fit (weighted by num_examples)
    # 3. Each round: configure_evaluate -> clients evaluate -> aggregate_evaluate (weighted by num_examples)
    # 4. Optional: evaluate_fn runs server-side evaluation on centralized test set
    # 5. Returns Result object with final model and metrics history
    logger.info("=" * 80)
    logger.info(f"STARTING FEDERATED LEARNING: {num_rounds} rounds")
    logger.info("Aggregation method: FedAvg with weighted averaging by num_examples")
    logger.info("Server evaluation: ENABLED (centralized test set)")
    logger.info("=" * 80)

    # Start federated learning
    logger.info("Starting federated learning...")
    try:
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
            evaluate_fn=central_evaluate_fn,  # Pass evaluate_fn to start() method
        )
        logger.info("Federated learning completed.")

        # Mark run as completed
        if run_id:
            logger.info("Marking federated run as completed in database...")
            db = get_session()
            try:
                run_crud.complete_run(db, run_id=run_id, status="completed")
                db.commit()
                logger.info(f"[OK] Run {run_id} marked as completed with end_time")
            except Exception as e:
                logger.error(f"[ERROR] Failed to update run completion: {e}")
                db.rollback()
            finally:
                db.close()

    except Exception as e:
        logger.error(f"Federated learning failed: {str(e)}", exc_info=True)

        # Mark run as failed
        if run_id:
            logger.error(f"Marking run {run_id} as failed in database...")
            db = get_session()
            try:
                run_crud.complete_run(db, run_id=run_id, status="failed")
                db.commit()
                logger.error(f"[OK] Run {run_id} marked as failed")
            except Exception as db_error:
                logger.error(f"[ERROR] Failed to mark run as failed: {db_error}")
                db.rollback()
            finally:
                db.close()

        raise  # Re-raise to propagate error

    all_results = {}

    if result.train_metrics_clientapp:
        all_results["train_metrics_clientapp"] = _convert_metric_record_to_dict(
            result.train_metrics_clientapp
        )

    if result.evaluate_metrics_clientapp:
        all_results["evaluate_metrics_clientapp"] = _convert_metric_record_to_dict(
            result.evaluate_metrics_clientapp
        )

    if result.evaluate_metrics_serverapp:
        all_results["evaluate_metrics_serverapp"] = _convert_metric_record_to_dict(
            result.evaluate_metrics_serverapp
        )

    # Save all results to JSON
    # Use analysis_run_id if provided (for analysis module compatibility)
    # Otherwise fall back to database run_id
    result_file_id = analysis_run_id if analysis_run_id else run_id
    with open(f"results_{result_file_id}.json", "w") as f:
        import json

        json.dump(all_results, f, indent=2)
    logger.info(f"[OK] Results saved to results_{result_file_id}.json")

    # Persist server evaluation metrics to database
    if result.evaluate_metrics_serverapp and run_id:
        logger.info("Persisting server evaluation metrics to database...")
        _persist_server_evaluations(run_id, result.evaluate_metrics_serverapp)
    else:
        logger.error(
            f"[WARN] Skipping server evaluation persistence: "
            f"evaluate_metrics_serverapp={bool(result.evaluate_metrics_serverapp)}, "
            f"run_id={run_id}"
        )

    # Send training_end event to frontend now that ALL rounds are complete
    logger.info(
        "All federated rounds complete. Sending training_end event to frontend..."
    )
    ws_sender.send_metrics(
        {
            "run_id": run_id,
            "status": "completed",
            "experiment_name": f"federated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_rounds": num_rounds,
            "training_mode": "federated",
        },
        "training_end",
    )
    logger.info(f"[OK] Training complete notification sent (run_id={run_id})")
