from datasets.utils.logging import disable_progress_bar
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
    CentralizedTrainer,
)
from federated_pneumonia_detection.src.control.federated_new_version.partioner import (
    CustomPartitioner,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import (
    XRayDataModule,
)
from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any

disable_progress_bar()

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    configs = ConfigLoader()
    constants = configs.create_system_constants()
    config = configs.create_experiment_config()
    centerlized_trainer = CentralizedTrainer(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml",
        results_dir=r"federated_pneumonia_detection\results",
    )
    configs = msg.content["configs"]
    full_dataset = pd.read_csv(configs["file_path"])
    # TODO: Partition the data based on the passed configuration
    partioner = CustomPartitioner(full_dataset, configs["num_partitions"])

    partion_df = partioner.load_partition(configs["partition_id"])

    train_df, val_df = train_test_split(partion_df, test_size=0.2, random_state=42)

    data_module = XRayDataModule(
        train_df=train_df,
        val_df=val_df,
        constants=constants,
        config=config,
        image_dir=configs["image_dir"],
    )

    model, callbacks, metrics_collector = (
        centerlized_trainer._build_model_and_callbacks(
            train_df=train_df,
            experiment_name="federated_pneumonia_detection",
            run_id=configs["run_id"],
        )
    )

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainer = centerlized_trainer._build_trainer(
        callbacks=callbacks,
        experiment_name="federated_pneumonia_detection",
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
    model = LitResNet.load_state_dict(msg.content["model"].to_torch_state_dict())
    centerlized_trainer = CentralizedTrainer(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml",
        results_dir=r"federated_pneumonia_detection\results",
    )
    configs = msg.content["configs"]

    train_df, val_df = centerlized_trainer._prepare_dataset(
        csv_path=configs["csv_path"],
        image_dir=configs["image_dir"],
    )
    data_module = centerlized_trainer._create_data_module(
        train_df=train_df, val_df=val_df, image_dir=configs["image_dir"]
    )
    val_loader = data_module.val_dataloader()

    model, callbacks, metrics_collector = (
        centerlized_trainer._build_model_and_callbacks(
            train_df=train_df,
            experiment_name="federated_pneumonia_detection",
            run_id=configs["run_id"],
        )
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainer = centerlized_trainer._build_trainer(
        callbacks=callbacks,
        experiment_name="federated_pneumonia_detection",
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
