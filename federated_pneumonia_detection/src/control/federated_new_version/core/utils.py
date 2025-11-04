from flwr.app import Context
from sklearn.model_selection import train_test_split
from typing import Any
import pandas as pd
from pathlib import Path

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
    CentralizedTrainer,
)
from federated_pneumonia_detection.src.control.federated_new_version.partioner import (
    CustomPartitioner,
)
from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.utils.loggers.logger import setup_logger
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from typing import Dict

logger = setup_logger(__name__)


def filter_list_of_dicts(data: list[dict[str, Any]], fields: list[str]):
    """
    Filters a list of dictionaries to include only specified keys.

    Args:
      data: A list of dictionaries.
      fields: A list of keys to keep in each dictionary.

    Returns:
      A new list of dictionaries with only the specified keys.
    """
    metrics = {}
    for metric in data:
        for k, v in metric.items():
            if k in fields:
                metrics.update({k: v})

    return metrics


def _load_trainer_and_config():
    """Initialize and return trainer with config."""
    centerlized_trainer = CentralizedTrainer(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml"
    )
    return centerlized_trainer, centerlized_trainer.config


def _get_partition_data(configs: dict):
    """Load and partition dataset based on configs."""
    full_dataset = pd.read_csv(configs["file_path"])
    partioner = CustomPartitioner(full_dataset, configs["num_partitions"])
    return full_dataset, partioner


def _prepare_partition_and_split(
    partioner: CustomPartitioner, partition_id: int, partion_df
):
    """Split partition into train and validation sets."""
    train_df, val_df = train_test_split(partion_df, test_size=0.2, random_state=42)
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
            f"[Utils] Building components for federated client_id={client_id}, round={round_number}, run_id={run_id}"
        )

    model, callbacks, metrics_collector = (
        centerlized_trainer._build_model_and_callbacks(
            train_df=train_df,
            experiment_name="federated_pneumonia_detection",
            run_id=run_id,  # Use passed run_id instead of context.run_id
            is_federated=is_federated,
            client_id=client_id,
            round_number=round_number,
        )
    )
    return model, callbacks, metrics_collector


def _build_trainer_component(
    centerlized_trainer: CentralizedTrainer,
    callbacks,
    is_federated: bool,
):
    """Build trainer with callbacks."""
    trainer = centerlized_trainer._build_trainer(
        callbacks=callbacks,
        experiment_name="federated_pneumonia_detection",
        is_federated=is_federated,
    )
    return trainer


def _prepare_evaluation_dataframe(df):
    """Add filename column if it doesn't exist."""
    if "filename" not in df.columns and "patientId" in df.columns:
        df["filename"] = df.apply(lambda x: str(x["patientId"]) + ".png", axis=1)
    return df


def _extract_metrics_from_result(result_dict: dict):
    """Extract and map metrics from result dictionary.

    Note: Uses 'is not None' check instead of 'or' to handle legitimate 0.0 values.
    Handles both 'test_acc' and 'test_accuracy' naming conventions.
    """
    loss = (
        result_dict.get("test_loss")
        if result_dict.get("test_loss") is not None
        else result_dict.get("loss", 0.0)
    )

    # Handle both test_acc and test_accuracy
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
    loss, accuracy, precision, recall, f1, auroc, num_examples: int
):
    """Create metric record dictionary with all metrics.

    Following Flower conventions:
    - 'num-examples' key (with HYPHEN) is used for weighted aggregation
    - This key must be present for FedAvg to properly weight client contributions
    """
    return {
        "test_loss": loss,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_auroc": auroc,
        "num-examples": num_examples,  # CRITICAL: Must be "num-examples" with HYPHEN!
    }


def read_configs_to_toml() -> dict:
    """Read federated learning configs from default_config.yaml and prepare for pyproject.toml.

    This function extracts Flower-specific configuration values from the YAML config
    and returns them in a format suitable for updating pyproject.toml.

    Returns:
        dict: Configuration dictionary with keys: num_server_rounds, max_epochs, num_supernodes

    Note:
        These configs correspond to:
        - num_server_rounds -> [tool.flwr.app.config] num-server-rounds
        - max_epochs -> [tool.flwr.app.config] max-epochs
        - num_supernodes -> [tool.flwr.federations.local-simulation.options] num-supernodes
    """
    config_dir = (
        Path(__file__).parent.parent.parent.parent.parent
        / "config"
        / "default_config.yaml"
    )

    print(f"[Config Reader] Reading from: {config_dir}")

    try:
        config_manager = ConfigManager(config_path=str(config_dir))
    except Exception as e:
        print(f"[Config Reader] ❌ Failed to load config: {e}")
        return {}

    flwr_configs = {}

    # Read num-server-rounds (number of federated learning rounds)
    if config_manager.has_key("experiment.num-server-rounds"):
        flwr_configs["num_server_rounds"] = config_manager.get(
            "experiment.num-server-rounds"
        )
        print(
            f"[Config Reader] ✓ num-server-rounds: {flwr_configs['num_server_rounds']}"
        )
    else:
        print("[Config Reader] ⚠️ experiment.num-server-rounds not found in config")

    # Read max-epochs (local training epochs per round)
    if config_manager.has_key("experiment.max-epochs"):
        flwr_configs["max_epochs"] = config_manager.get("experiment.max-epochs")
        print(f"[Config Reader] ✓ max-epochs: {flwr_configs['max_epochs']}")
    elif config_manager.has_key("experiment.local_epochs"):
        # Fallback to local_epochs if max-epochs not found
        flwr_configs["max_epochs"] = config_manager.get("experiment.local_epochs")
        print(
            f"[Config Reader] ✓ max-epochs (from local_epochs): {flwr_configs['max_epochs']}"
        )
    else:
        print(
            "[Config Reader] ⚠️ experiment.max-epochs or experiment.local_epochs not found"
        )

    # Read num-supernodes (number of clients in simulation)
    if config_manager.has_key("experiment.options.num-supernodes"):
        flwr_configs["num_supernodes"] = config_manager.get(
            "experiment.options.num-supernodes"
        )
        print(f"[Config Reader] ✓ num-supernodes: {flwr_configs['num_supernodes']}")
    elif config_manager.has_key("experiment.num_clients"):
        # Fallback to num_clients if num-supernodes not found
        flwr_configs["num_supernodes"] = config_manager.get("experiment.num_clients")
        print(
            f"[Config Reader] ✓ num-supernodes (from num_clients): {flwr_configs['num_supernodes']}"
        )
    else:
        print(
            "[Config Reader] ⚠️ experiment.options.num-supernodes or experiment.num_clients not found"
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
    """
    Persist server-side evaluation metrics to database.

    Args:
        run_id: Database run ID
        server_metrics: Dictionary mapping round number to MetricRecord
    """
    # Verify database connection before attempting persistence
    logger.info("=" * 80)
    logger.info("DATABASE PERSISTENCE - Server Evaluations")
    logger.info("=" * 80)

    try:
        # Test database connection first
        from federated_pneumonia_detection.config.settings import Settings

        settings_obj = Settings()
        db_uri = settings_obj.get_postgres_db_uri()
        logger.info(f"Database URI configured: {db_uri[:20]}... (truncated)")
    except Exception as e:
        logger.error(
            f"❌ CRITICAL: Failed to load database settings: {e}", exc_info=True
        )
        logger.error("Check if .env file is loaded and environment variables are set!")
        return

    db = None
    try:
        db = get_session()
        logger.info(f"✅ Database session created successfully")
        logger.info(f"Processing {len(server_metrics)} server evaluation rounds...")

        for round_num_str, metric_record in server_metrics.items():
            # Convert round number to int (Flower uses int keys, but may be converted to string)
            round_num = int(round_num_str)
            logger.info(f"  Processing round {round_num} (key={round_num_str})...")

            # Convert MetricRecord to dict if needed
            if hasattr(metric_record, "__dict__"):
                metrics_dict = dict(metric_record)
            elif isinstance(metric_record, str):
                # If it's a string representation, parse it
                import ast

                try:
                    metrics_dict = ast.literal_eval(metric_record)
                    logger.info(
                        f"    Parsed string metrics: {list(metrics_dict.keys())}"
                    )
                except Exception as parse_err:
                    logger.error(f"    Failed to parse metric string: {parse_err}")
                    metrics_dict = {}
            else:
                metrics_dict = metric_record

            logger.info(
                f"    Metrics dict type: {type(metrics_dict)}, keys: {list(metrics_dict.keys()) if isinstance(metrics_dict, dict) else 'N/A'}"
            )

            # Extract metrics with 'server_' prefix (from server_evaluation.py)
            extracted_metrics = {
                "loss": metrics_dict.get("server_loss", 0.0),
                "accuracy": metrics_dict.get("server_accuracy"),
                "precision": metrics_dict.get("server_precision"),
                "recall": metrics_dict.get("server_recall"),
                "f1_score": metrics_dict.get("server_f1"),
                "auroc": metrics_dict.get("server_auroc"),
            }

            logger.info(f"    Extracted metrics: {extracted_metrics}")

            # Create server evaluation record
            server_evaluation_crud.create_evaluation(
                db=db,
                run_id=run_id,
                round_number=round_num,
                metrics=extracted_metrics,
                num_samples=metrics_dict.get("num_samples"),
            )
            logger.info(f"  ✅ Persisted server evaluation for round {round_num}")

        db.commit()
        logger.info("=" * 80)
        logger.info(f"✅ SUCCESS: Persisted {len(server_metrics)} server evaluations")
        logger.info("=" * 80)
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ CRITICAL ERROR: Failed to persist server evaluations")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}", exc_info=True)
        logger.error("=" * 80)
        if db:
            db.rollback()
    finally:
        if db:
            db.close()
            logger.info("Database session closed")
