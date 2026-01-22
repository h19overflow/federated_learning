import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from flwr.app import ArrayRecord, Context
from sklearn.model_selection import train_test_split

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
    CentralizedTrainer,
)
from federated_pneumonia_detection.src.control.dl_model.centralized_trainer_utils.model_setup import (  # noqa: E501
    build_model_and_callbacks,
    build_trainer,
)
from federated_pneumonia_detection.src.control.dl_model.internals.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
)
from federated_pneumonia_detection.src.control.federated_new_version.core.custom_strategy import (
    ConfigurableFedAvg,
)
from federated_pneumonia_detection.src.control.federated_new_version.partioner import (
    CustomPartitioner,
)
from federated_pneumonia_detection.src.internals.loggers.logger import setup_logger

logger = setup_logger(__name__)


def _initialize_database_run() -> Tuple[int | None, Any]:
    """Initialize federated training run in database.
    
    Returns:
        Tuple of (run_id, db_session) or (None, None) on failure
    """
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
        new_run = run_crud.create(db, **run_data)
        db.commit()
        run_id = new_run.id
        logger.info(f"[OK] Successfully created run with id={run_id}")
        return run_id, db
    except Exception as e:
        logger.error(f"[ERROR] Failed to create run in database: {e}", exc_info=True)
        db.rollback()
        db.close()
        return None, None


def _setup_config_manager() -> Tuple[ConfigManager, str, str]:
    """Setup configuration manager with environment overrides.
    
    Returns:
        Tuple of (config_manager, experiment_seed, analysis_run_id)
    """
    default_config_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "config"
        / "default_config.yaml"
    )
    config_path = os.getenv("CONFIG_PATH", str(default_config_path))
    config_manager = ConfigManager(config_path=str(config_path))

    fl_seed = os.getenv("FL_SEED")
    if fl_seed:
        seed_value = int(fl_seed)
        config_manager.set("experiment.seed", seed_value)
        config_manager.set("system.seed", seed_value)
        logger.info(f"[ENV OVERRIDE] Using seed from FL_SEED: {seed_value}")

    experiment_seed = config_manager.get("experiment.seed", 42)
    analysis_run_id = os.getenv("FL_RUN_ID")
    if analysis_run_id:
        logger.info(f"[ENV OVERRIDE] Using run_id from FL_RUN_ID: {analysis_run_id}")

    return config_manager, experiment_seed, analysis_run_id


def _build_training_configs(
    config_manager: ConfigManager,
    num_clients: int,
    run_id: int | None,
    experiment_seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build training and evaluation configs to send to clients.
    
    Args:
        config_manager: Configuration manager instance
        num_clients: Number of clients (for num_partitions)
        run_id: Database run ID
        experiment_seed: Experiment seed value
        
    Returns:
        Tuple of (train_config, eval_config)
    """
    train_config = {
        "file_path": config_manager.get("experiment.file-path"),
        "image_dir": config_manager.get("experiment.image-dir"),
        "num_partitions": num_clients,
        "run_id": run_id,
        "seed": experiment_seed,
    }

    eval_config = {
        "csv_path": config_manager.get("experiment.file-path"),
        "image_dir": config_manager.get("experiment.image-dir"),
    }

    return train_config, eval_config


def _build_global_model(config_manager: ConfigManager) -> Tuple[LitResNetEnhanced, ArrayRecord]:
    """Build and initialize global model for federated learning.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Tuple of (model, arrays_record)
    """
    num_classes = 2
    class_weights = torch.ones(num_classes)

    global_model = LitResNetEnhanced(
        config=config_manager,
        class_weights_tensor=class_weights,
        use_focal_loss=True,
        focal_alpha=0.6,
        focal_gamma=1.5,
        label_smoothing=0.05,
        use_cosine_scheduler=True,
        monitor_metric="val_recall",
    )
    logger.info("Initialized LitResNetEnhanced with balanced focal loss configuration")

    arrays = ArrayRecord(global_model.state_dict())
    return global_model, arrays


def _initialize_websocket_sender(
    num_clients: int,
    num_rounds: int,
) -> MetricsWebSocketSender:
    """Initialize WebSocket sender and broadcast training mode.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of rounds
        
    Returns:
        MetricsWebSocketSender instance
    """
    logger.info("Initializing WebSocket sender for real-time metrics...")
    ws_sender = MetricsWebSocketSender("ws://localhost:8765")

    logger.info(
        f"Broadcasting training mode: {num_clients} clients, {num_rounds} rounds",
    )
    ws_sender.send_training_mode(
        is_federated=True,
        num_rounds=num_rounds,
        num_clients=num_clients,
    )

    return ws_sender


def _initialize_strategy(
    train_config: Dict[str, Any],
    eval_config: Dict[str, Any],
    run_id: int | None,
    num_rounds: int,
) -> ConfigurableFedAvg:
    """Initialize FedAvg strategy with configurations.
    
    Args:
        train_config: Training configuration
        eval_config: Evaluation configuration
        run_id: Database run ID
        num_rounds: Number of rounds
        
    Returns:
        Configured strategy instance
    """
    logger.info("Initializing FedAvg strategy...")
    strategy = ConfigurableFedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        train_config=train_config,
        eval_config=eval_config,
        weighted_by_key="num-examples",
        websocket_uri="ws://localhost:8765",
        run_id=run_id,
    )

    strategy.set_total_rounds(num_rounds)
    logger.info(
        "Strategy configured: FedAvg with weighted aggregation by num_examples + server evaluation",
    )

    return strategy


def filter_list_of_dicts(data: list[dict[str, Any]], fields: list[str]):
    """Filter list of dicts to include only specified keys from last epoch."""
    DEFAULT_FLOAT = 0.0
    DEFAULT_INT = 0

    if not data:
        out: dict[str, Any] = {}
        for field in fields:
            out[field] = DEFAULT_INT if field == "epoch" else DEFAULT_FLOAT
        return out

    last_epoch_metrics = data[-1]

    metrics: dict[str, Any] = {}
    for field in fields:
        val = last_epoch_metrics.get(field)
        if field == "epoch":
            if val is None:
                metrics[field] = DEFAULT_INT
            else:
                try:
                    metrics[field] = int(val)
                except Exception:
                    metrics[field] = DEFAULT_INT
            continue

        if val is None:
            metrics[field] = DEFAULT_FLOAT
            continue

        try:
            metrics[field] = float(val)
        except Exception:
            metrics[field] = DEFAULT_FLOAT

    return metrics


def _load_trainer_and_config():
    """Initialize and return trainer with config."""
    centerlized_trainer = CentralizedTrainer(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml",
    )
    return centerlized_trainer, centerlized_trainer.config


def _get_partition_data(configs: dict):
    """Load and partition dataset based on configs."""
    full_dataset = pd.read_csv(configs["file_path"])
    partioner = CustomPartitioner(full_dataset, configs["num_partitions"])
    return full_dataset, partioner


def _prepare_partition_and_split(
    partioner: CustomPartitioner,
    partition_id: int,
    partion_df,
    seed: int = 42,
):
    """Split partition into train and validation sets."""
    train_df, val_df = train_test_split(partion_df, test_size=0.2, random_state=seed)
    return train_df, val_df


def _build_model_components(
    centerlized_trainer: CentralizedTrainer,
    train_df,
    context: Context,
    is_federated: bool,
    client_id: int = None,
    round_number: int = 0,
    run_id: int = None,
):
    """Build model, callbacks, and metrics collector with federated context."""
    if is_federated and client_id is not None:
        centerlized_trainer.logger.info(
            f"[Utils] Building components for federated client_id={client_id}, "  # noqa: E501
            f"round={round_number}, run_id={run_id}",
        )

    model, callbacks, metrics_collector = build_model_and_callbacks(
        train_df=train_df,
        config=centerlized_trainer.config,
        checkpoint_dir=centerlized_trainer.checkpoint_dir,
        logs_dir=centerlized_trainer.logs_dir,
        logger=centerlized_trainer.logger,
        experiment_name="federated_pneumonia_detection",
        run_id=run_id,
        is_federated=is_federated,
        client_id=client_id,
        round_number=round_number,
    )
    return model, callbacks, metrics_collector


def _build_trainer_component(
    centerlized_trainer: CentralizedTrainer,
    callbacks,
    is_federated: bool,
):
    """Build trainer with callbacks."""
    trainer = build_trainer(
        config=centerlized_trainer.config,
        callbacks=callbacks,
        logs_dir=centerlized_trainer.logs_dir,
        experiment_name="federated_pneumonia_detection",
        logger=centerlized_trainer.logger,
        is_federated=is_federated,
    )
    return trainer


def _prepare_evaluation_dataframe(df):
    """Add filename column if it doesn't exist."""
    if "filename" not in df.columns and "patientId" in df.columns:
        df["filename"] = df.apply(lambda x: str(x["patientId"]) + ".png", axis=1)
    return df


def _extract_metrics_from_result(result_dict: dict):
    """Extract and map metrics from result dictionary."""
    loss = (
        result_dict.get("test_loss")
        if result_dict.get("test_loss") is not None
        else result_dict.get("loss", 0.0)
    )

    accuracy = result_dict.get("test_accuracy")
    if accuracy is None:
        accuracy = result_dict.get("test_acc")
    if accuracy is None:
        accuracy = result_dict.get("accuracy", 0.0)

    precision = (
        result_dict.get("test_precision")
        if result_dict.get("test_precision") is not None
        else result_dict.get("precision", 0.0)
    )
    recall = (
        result_dict.get("test_recall")
        if result_dict.get("test_recall") is not None
        else result_dict.get("recall", 0.0)
    )
    f1 = (
        result_dict.get("test_f1")
        if result_dict.get("test_f1") is not None
        else result_dict.get("f1", 0.0)
    )
    auroc = (
        result_dict.get("test_auroc")
        if result_dict.get("test_auroc") is not None
        else result_dict.get("auroc", 0.0)
    )
    return loss, accuracy, precision, recall, f1, auroc


def _create_metric_record_dict(
    loss,
    accuracy,
    precision,
    recall,
    f1,
    auroc,
    num_examples: int,
):
    """Create metric record dictionary with all metrics."""
    return {
        "test_loss": loss,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_auroc": auroc,
        "num-examples": num_examples,
    }


def read_configs_to_toml() -> dict:
    """Read federated learning configs from default_config.yaml."""
    config_dir = (
        Path(__file__).parent.parent.parent.parent.parent
        / "config"
        / "default_config.yaml"
    )

    print(f"[Config Reader] Reading from: {config_dir}")

    try:
        config_manager = ConfigManager(config_path=str(config_dir))
    except Exception as e:
        print(f"[Config Reader] [ERROR] Failed to load config: {e}")
        return {}

    flwr_configs = {}

    if config_manager.has_key("experiment.num-server-rounds"):
        flwr_configs["num_server_rounds"] = config_manager.get(
            "experiment.num-server-rounds",
        )
        print(
            f"[Config Reader] [OK] num-server-rounds: "
            f"{flwr_configs['num_server_rounds']}",
        )
    else:
        print("[Config Reader] [WARN] experiment.num-server-rounds not found in config")

    if config_manager.has_key("experiment.max-epochs"):
        flwr_configs["max_epochs"] = config_manager.get("experiment.max-epochs")
        print(f"[Config Reader] [OK] max-epochs: {flwr_configs['max_epochs']}")
    elif config_manager.has_key("experiment.local_epochs"):
        flwr_configs["max_epochs"] = config_manager.get("experiment.local_epochs")
        print(
            f"[Config Reader] [OK] max-epochs (from local_epochs): "
            f"{flwr_configs['max_epochs']}",
        )
    else:
        print(
            "[Config Reader] [WARN] experiment.max-epochs or "
            "experiment.local_epochs not found",
        )

    if config_manager.has_key("experiment.options.num-supernodes"):
        flwr_configs["num_supernodes"] = config_manager.get(
            "experiment.options.num-supernodes",
        )
        print(f"[Config Reader] [OK] num-supernodes: {flwr_configs['num_supernodes']}")
    elif config_manager.has_key("experiment.num_clients"):
        flwr_configs["num_supernodes"] = config_manager.get("experiment.num_clients")
        print(
            f"[Config Reader] [OK] num-supernodes (from num_clients): "
            f"{flwr_configs['num_supernodes']}",
        )
    else:
        print(
            "[Config Reader] [WARN] experiment.options.num-supernodes or "
            "experiment.num_clients not found",
        )

    print(f"[Config Reader] Final configs to write to pyproject.toml: {flwr_configs}")
    return flwr_configs


def _convert_metric_record_to_dict(data):
    """Convert MetricRecord objects and nested structures to plain dicts/lists."""
    if isinstance(data, dict):
        return {str(k): _convert_metric_record_to_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_metric_record_to_dict(item) for item in data]
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        return str(data)


def _persist_server_evaluations(run_id: int, server_metrics: Dict[int, Any]) -> None:
    """Persist server-side evaluation metrics to database."""
    logger.info("=" * 80)
    logger.info("DATABASE PERSISTENCE - Server Evaluations")
    logger.info("=" * 80)

    try:
        from federated_pneumonia_detection.config.settings import Settings

        settings_obj = Settings()
        db_uri = settings_obj.get_postgres_db_uri()
        logger.info(f"Database URI configured: {db_uri[:20]}... (truncated)")
    except Exception as e:
        logger.error(
            f"[ERROR] CRITICAL: Failed to load database settings: {e}",
            exc_info=True,
        )
        logger.error("Check if .env file is loaded and environment variables are set!")
        return

    db = None
    try:
        db = get_session()
        logger.info("[OK] Database session created successfully")
        logger.info(f"Processing {len(server_metrics)} server evaluation rounds...")

        for round_num_str, metric_record in server_metrics.items():
            round_num = int(round_num_str)
            logger.info(f"  Processing round {round_num} (key={round_num_str})...")

            if hasattr(metric_record, "__dict__"):
                metrics_dict = dict(metric_record)
            elif isinstance(metric_record, str):
                import ast

                try:
                    metrics_dict = ast.literal_eval(metric_record)
                    logger.info(
                        f"    Parsed string metrics: {list(metrics_dict.keys())}",
                    )
                except Exception as parse_err:
                    logger.error(f"    Failed to parse metric string: {parse_err}")
                    metrics_dict = {}
            else:
                metrics_dict = metric_record

            keys_str = (
                list(metrics_dict.keys()) if isinstance(metrics_dict, dict) else "N/A"
            )
            logger.info(
                f"    Metrics dict type: {type(metrics_dict)}, keys: {keys_str}",
            )

            extracted_metrics = {
                "loss": metrics_dict.get("server_loss", 0.0),
                "accuracy": metrics_dict.get("server_accuracy"),
                "precision": metrics_dict.get("server_precision"),
                "recall": metrics_dict.get("server_recall"),
                "f1_score": metrics_dict.get("server_f1"),
                "auroc": metrics_dict.get("server_auroc"),
            }

            logger.info(f"    Extracted metrics: {extracted_metrics}")

            server_evaluation_crud.create_evaluation(
                db=db,
                run_id=run_id,
                round_number=round_num,
                metrics=extracted_metrics,
                num_samples=metrics_dict.get("num_samples"),
            )
            logger.info(f"  [OK] Persisted server evaluation for round {round_num}")

        db.commit()
        logger.info("=" * 80)
        logger.info(f"[OK] SUCCESS: Persisted {len(server_metrics)} server evaluations")
        logger.info("=" * 80)
    except Exception as e:
        logger.error("=" * 80)
        logger.error("[ERROR] CRITICAL ERROR: Failed to persist server evaluations")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}", exc_info=True)
        logger.error("=" * 80)
        if db:
            db.rollback()
    finally:
        if db:
            db.close()
            logger.info("Database session closed")
