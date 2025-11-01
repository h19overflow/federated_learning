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
    train_configs = msg.content.get("configs", {
        "file_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
        "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
        "num_partitions": 4,
        "partition_id": 0,
        "run_id": 5,
    })
    configs = train_configs
    full_dataset = pd.read_csv(configs["file_path"])
    partioner = CustomPartitioner(full_dataset, configs["num_partitions"])

    partion_df = partioner.load_partition(configs["partition_id"])
    
    # Add filename column if it doesn't exist (required by XRayDataModule)
    if 'filename' not in partion_df.columns and 'patientId' in partion_df.columns:
        partion_df['filename'] = partion_df.apply(lambda x: str(x['patientId']) + '.png', axis=1)

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
            run_id=configs["run_id"],
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
    config = msg.content.get("configs", {})
    current_round = config["server-round"]  # Gets the current round number
    current_client_id = context.node.id
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
    if os.path.exists(os.path.join(config.logs_dir, 'metrics_output')):
        with open(os.path.join(config.logs_dir, 'metrics_output', f'metrics_{current_round}_{current_client_id}.json'), 'w') as f:
            json.dump(metrics_history, f)
    else: 
        os.makedirs(os.path.join(config.logs_dir, 'metrics_output'), exist_ok=True)
        with open(os.path.join(config.logs_dir, 'metrics_output', f'metrics_{current_round}_{current_client_id}.json'), 'w') as f:
            json.dump(metrics_history, f)
    print(f"Metrics saved to: {os.path.join(config.logs_dir, 'metrics_output', 'metrics.json')}")
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    centerlized_trainer = CentralizedTrainer(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml",
    )
    # Get configs from message (safely handle missing key with defaults)
    eval_configs = msg.content.get("configs", {
        "csv_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
        "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
        "run_id": 5,
    })
    configs = eval_configs

    train_df, val_df = centerlized_trainer._prepare_dataset(
        csv_path=configs["csv_path"],
        image_dir=configs["image_dir"],
    )
    
    # Add filename column if it doesn't exist (required by XRayDataModule)
    if 'filename' not in train_df.columns and 'patientId' in train_df.columns:
        train_df['filename'] = train_df.apply(lambda x: str(x['patientId']) + '.png', axis=1)
    if 'filename' not in val_df.columns and 'patientId' in val_df.columns:
        val_df['filename'] = val_df.apply(lambda x: str(x['patientId']) + '.png', axis=1)
    
    data_module = centerlized_trainer._create_data_module(
        train_df=train_df, val_df=val_df, image_dir=configs["image_dir"]
    )
    val_loader = data_module.val_dataloader()

    model, callbacks, metrics_collector = (
        centerlized_trainer._build_model_and_callbacks(
            train_df=train_df,
            experiment_name="federated_pneumonia_detection",
            run_id=configs["run_id"],
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
    loss = results[0]["test_loss"]
    accuracy = results[0]["test_accuracy"]
    precision = results[0]["test_precision"]
    recall = results[0]["test_recall"]
    f1 = results[0]["test_f1"]
    auroc = results[0]["test_auroc"]
    metric_record = MetricRecord(
        {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_auroc": auroc,
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
