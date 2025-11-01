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
):
    """Build model, callbacks, and metrics collector."""
    model, callbacks, metrics_collector = (
        centerlized_trainer._build_model_and_callbacks(
            train_df=train_df,
            experiment_name="federated_pneumonia_detection",
            run_id=context.run_id,
            is_federated=is_federated,
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
    """Extract and map metrics from result dictionary."""
    loss = result_dict.get("test_loss") or result_dict.get("loss", 0.0)
    accuracy = result_dict.get("test_accuracy") or result_dict.get("accuracy", 0.0)
    precision = result_dict.get("test_precision") or result_dict.get("precision", 0.0)
    recall = result_dict.get("test_recall") or result_dict.get("recall", 0.0)
    f1 = result_dict.get("test_f1") or result_dict.get("f1", 0.0)
    auroc = result_dict.get("test_auroc") or result_dict.get("auroc", 0.0)
    return loss, accuracy, precision, recall, f1, auroc


def _create_metric_record_dict(
    loss, accuracy, precision, recall, f1, auroc, num_examples: int
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
    """Read and convert YAML config to dictionary format."""
    config_dir = Path(__file__).parent.parent.parent.parent.parent / "config" / "default_config.yaml"
    config_manager = ConfigManager(config_path=str(config_dir))
    flwr_configs = {}
    if config_manager.has_key("experiment.num-server-rounds"):
        flwr_configs["num_server_rounds"] = config_manager.get(
            "experiment.num-server-rounds"
        )
    if config_manager.has_key("experiment.max-epochs"):
        flwr_configs["max_epochs"] = config_manager.get("experiment.max-epochs")
    if config_manager.has_key("experiment.options.num-supernodes"):
        flwr_configs["num_supernodes"] = config_manager.get(
            "experiment.options.num-supernodes"
        )
    else:
        print("No num-supernodes found in config")
    print(f"Loaded flwr_configs: {flwr_configs}")
    return flwr_configs

