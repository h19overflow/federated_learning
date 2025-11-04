from datasets.utils.logging import disable_progress_bar
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import (
    XRayDataModule,
)
import pandas as pd
from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
    filter_list_of_dicts,
    _load_trainer_and_config,
    _get_partition_data,
    _prepare_partition_and_split,
    _build_model_components,
    _build_trainer_component,
    _prepare_evaluation_dataframe,
    _extract_metrics_from_result,
    _create_metric_record_dict,
)

disable_progress_bar()

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    # Initialize trainer and config
    centerlized_trainer, config = _load_trainer_and_config()

    # Extract client_id and round_number from Flower context for federated metrics tracking
    client_id = context.node_id
    round_number = (
        context.state.current_round if hasattr(context.state, "current_round") else 0
    )
    centerlized_trainer.logger.info(
        f"[Federated Train] Starting training for client_id={client_id}, round={round_number}"
    )

    # Get configs from message (safely handle missing key with defaults)
    configs = msg.content.get(
        "config",
        {
            "file_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
            "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
            "num_partitions": 2,
        },
    )

    # Load and partition dataset
    _, partioner = _get_partition_data(configs)
    partition_id = context.node_id % configs["num_partitions"]
    partion_df = partioner.load_partition(partition_id)
    train_df, val_df = _prepare_partition_and_split(partioner, partition_id, partion_df)

    # Create data module
    data_module = XRayDataModule(
        train_df=train_df,
        val_df=val_df,
        config=config,
        image_dir=configs["image_dir"],
    )

    # Extract run_id from configs (sent by server)
    run_id = configs.get("run_id", None)
    centerlized_trainer.logger.info(
        f"[Federated Train] Using run_id={run_id} from server config"
    )

    # Build model and trainer with client_id and round_number for federated context
    model, callbacks, metrics_collector = _build_model_components(
        centerlized_trainer,
        train_df,
        context,
        is_federated=True,
        client_id=client_id,
        round_number=round_number,
        run_id=run_id,  # Pass run_id from server config
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainer = _build_trainer_component(
        centerlized_trainer, callbacks, is_federated=True
    )

    # Train model and collect results
    trainer.fit(model, data_module)
    results = centerlized_trainer._collect_training_results(
        trainer=trainer,
        model=model,
        metrics_collector=metrics_collector,
    )

    # Filter and prepare metrics
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

    # Create and return response
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
    centerlized_trainer, _ = _load_trainer_and_config()

    # Get configs from message (safely handle missing key with defaults)
    eval_configs = msg.content.get(
        "config",
        {
            "csv_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
            "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
        },
    )

    configs = eval_configs
    # Prepare dataset
    train_df, val_df = centerlized_trainer._prepare_dataset(
        csv_path=configs["csv_path"],
        image_dir=configs["image_dir"],
    )

    # Add filename column if needed
    train_df = _prepare_evaluation_dataframe(train_df)
    val_df = _prepare_evaluation_dataframe(val_df)

    # Create data module and validator
    data_module = centerlized_trainer._create_data_module(
        train_df=train_df, val_df=val_df, image_dir=configs["image_dir"]
    )
    data_module.setup(stage="validate")
    val_loader = data_module.val_dataloader()

    # Build model and trainer
    model, callbacks, metrics_collector = _build_model_components(
        centerlized_trainer, train_df, context, is_federated=False
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainer = _build_trainer_component(
        centerlized_trainer, callbacks, is_federated=False
    )

    # Evaluate and extract metrics
    results = trainer.test(model, val_loader)
    result_dict = results[0] if results else {}
    loss, accuracy, precision, recall, f1, auroc = _extract_metrics_from_result(
        result_dict
    )

    # Create metric record
    metric_dict = _create_metric_record_dict(
        loss, accuracy, precision, recall, f1, auroc, len(val_df)
    )
    metric_record = MetricRecord(metric_dict)
    content = RecordDict(
        {
            "metrics": metric_record,
        }
    )
    return Message(content=content, reply_to=msg)
