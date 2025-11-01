from datasets.utils.logging import disable_progress_bar
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
    CentralizedTrainer,
)
from federated_pneumonia_detection.src.control.federated_new_version.partioner import (
    CustomPartitioner,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import (
    XRayDataModule,
)
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any
import os
import json

disable_progress_bar()

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    # Initialize trainer with config
    centerlized_trainer = CentralizedTrainer(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml"
    )
    config = centerlized_trainer.config

    # Get configs from message (safely handle missing key with defaults)
    configs = msg.content.get(
        "config",
        {
            "file_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
            "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
            "num_partitions": 2,
        },
    )

    full_dataset = pd.read_csv(configs["file_path"])
    partioner = CustomPartitioner(full_dataset, configs["num_partitions"])

    partition_id = context.node_id % configs["num_partitions"]
    partion_df = partioner.load_partition(partition_id)

    train_df, val_df = train_test_split(partion_df, test_size=0.2, random_state=42)

    data_module = XRayDataModule(
        train_df=train_df,
        val_df=val_df,
        config=config,
        image_dir=configs["image_dir"],
    )

    model, callbacks, metrics_collector = (
        centerlized_trainer._build_model_and_callbacks(
            train_df=train_df,
            experiment_name="federated_pneumonia_detection",
            run_id=context.run_id,
            is_federated=True,
        )
    )

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainer = centerlized_trainer._build_trainer(
        callbacks=callbacks,
        experiment_name="federated_pneumonia_detection",
        is_federated=True,
    )
    trainer.fit(model, data_module)

    results = centerlized_trainer._collect_training_results(
        trainer=trainer,
        model=model,
        metrics_collector=metrics_collector,
    )
    metrics_history = filter_list_of_dicts(
        results["metrics_history"],
        [
            "epoch",
            "train_loss",
            "train_acc",
            "train_f1",
            "val_loss",
            "val_acc",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_auroc",
        ],
    )
    metrics_history["num_examples"] = len(train_df)
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(metrics_history)
    content = RecordDict(
        {
            "model": model_record,
            "metrics": metric_record,
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    centerlized_trainer = CentralizedTrainer(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml",
    )
    # Get configs from message (safely handle missing key with defaults)
    eval_configs = msg.content.get(
        "config",
        {
            "csv_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
            "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
        },
    )
    configs = eval_configs

    train_df, val_df = centerlized_trainer._prepare_dataset(
        csv_path=configs["csv_path"],
        image_dir=configs["image_dir"],
    )

    # Add filename column if it doesn't exist (required by XRayDataModule)
    if "filename" not in train_df.columns and "patientId" in train_df.columns:
        train_df["filename"] = train_df.apply(
            lambda x: str(x["patientId"]) + ".png", axis=1
        )
    if "filename" not in val_df.columns and "patientId" in val_df.columns:
        val_df["filename"] = val_df.apply(
            lambda x: str(x["patientId"]) + ".png", axis=1
        )

    data_module = centerlized_trainer._create_data_module(
        train_df=train_df, val_df=val_df, image_dir=configs["image_dir"]
    )
    data_module.setup(stage="validate")
    val_loader = data_module.val_dataloader()

    model, callbacks, metrics_collector = (
        centerlized_trainer._build_model_and_callbacks(
            train_df=train_df,
            experiment_name="federated_pneumonia_detection",
            run_id=context.run_id,
            is_federated=False,
        )
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainer = centerlized_trainer._build_trainer(
        callbacks=callbacks,
        experiment_name="federated_pneumonia_detection",
        is_federated=False,
    )
    results = trainer.test(model, val_loader)
    result_dict = results[0] if results else {}

    loss = result_dict.get("test_loss") or result_dict.get("loss", 0.0)
    accuracy = result_dict.get("test_accuracy") or result_dict.get("accuracy", 0.0)
    precision = result_dict.get("test_precision") or result_dict.get("precision", 0.0)
    recall = result_dict.get("test_recall") or result_dict.get("recall", 0.0)
    f1 = result_dict.get("test_f1") or result_dict.get("f1", 0.0)
    auroc = result_dict.get("test_auroc") or result_dict.get("auroc", 0.0)
    metric_record = MetricRecord(
        {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_auroc": auroc,
            "num-examples": len(val_df),
        }
    )
    content = RecordDict(
        {
            "metrics": metric_record,
        }
    )
    return Message(content=content, reply_to=msg)


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
