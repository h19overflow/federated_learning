import random

import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer_utils import (  # noqa: E501
    collect_training_results,
    create_data_module,
    prepare_dataset,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.xray_data_module import (  # noqa: E501
    XRayDataModule,
)
from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
    _build_model_components,
    _build_trainer_component,
    _create_metric_record_dict,
    _extract_metrics_from_result,
    _get_partition_data,
    _load_trainer_and_config,
    _prepare_evaluation_dataframe,
    _prepare_partition_and_split,
    filter_list_of_dicts,
)

disable_progress_bar()

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on client data.

    Following Flower conventions:
    - Returns model updates (ArrayRecord) and metrics (MetricRecord)
    - Includes num_examples in metrics for proper weighted aggregation
    """
    centerlized_trainer, config = _load_trainer_and_config()

    client_id = context.node_id
    round_number = (
        context.state.current_round if hasattr(context.state, "current_round") else 0
    )
    centerlized_trainer.logger.info(
        f"[Federated Train] Starting training for client_id={client_id}, "
        f"round={round_number}",
    )

    configs = msg.content.get(
        "config",
        {
            "file_path": (  # noqa: E501
                r"C:\Users\User\Projects\FYP2\Training_Sample_5pct"
                r"\stage2_train_metadata.csv"
            ),
            "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
            "num_partitions": 2,
        },
    )

    _, partioner = _get_partition_data(configs)
    partition_id = context.node_id % configs["num_partitions"]
    partion_df = partioner.load_partition(partition_id)

    seed = configs.get("seed", 42)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    centerlized_trainer.logger.info(
        f"[Federated Train] Set global seed={seed} for reproducibility",
    )

    train_df, val_df = _prepare_partition_and_split(
        partioner,
        partition_id,
        partion_df,
        seed=seed,
    )

    data_module = XRayDataModule(
        train_df=train_df,
        val_df=val_df,
        config=config,
        image_dir=configs["image_dir"],
    )

    run_id = configs.get("run_id", None)
    centerlized_trainer.logger.info(
        f"[Federated Train] Using run_id={run_id} from server config",
    )

    model, callbacks, metrics_collector = _build_model_components(
        centerlized_trainer,
        train_df,
        context,
        is_federated=True,
        client_id=client_id,
        round_number=round_number,
        run_id=run_id,
    )

    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state_dict)

    first_param_name = list(model.state_dict().keys())[0]
    first_param_before = model.state_dict()[first_param_name].clone()
    centerlized_trainer.logger.info(
        f"[Client Train] BEFORE training - first param '{first_param_name}' "  # noqa: E501
        f"mean: {first_param_before.mean().item():.6f}",
    )

    trainer = _build_trainer_component(
        centerlized_trainer,
        callbacks,
        is_federated=True,
    )

    trainer.fit(model, data_module)

    first_param_after = model.state_dict()[first_param_name]
    centerlized_trainer.logger.info(
        f"[Client Train] AFTER training - first param '{first_param_name}' "  # noqa: E501
        f"mean: {first_param_after.mean().item():.6f}",
    )
    centerlized_trainer.logger.info(
        f"[Client Train] Parameter change: "  # noqa: E501
        f"{(first_param_after - first_param_before).abs().mean().item():.6f}",
    )
    results = collect_training_results(
        trainer=trainer,
        model=model,
        metrics_collector=metrics_collector,
        logs_dir=config.get("output.log_dir"),
        checkpoint_dir=config.get("output.checkpoint_dir"),
        logger=centerlized_trainer.logger,
        run_id=run_id,
    )

    num_examples = len(train_df)

    metrics_history = filter_list_of_dicts(
        results.get("metrics_history", []),
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

    metrics_history["num-examples"] = int(num_examples)

    centerlized_trainer.logger.info(
        f"[Client Train] Completed training with {num_examples} examples",
    )

    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(metrics_history)
    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metric_record,
        },
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the global model on client's local validation set.

    Following Flower conventions:
    - Returns evaluation metrics (MetricRecord)
    - Includes num_examples for proper weighted aggregation of metrics
    """
    centerlized_trainer, config = _load_trainer_and_config()

    client_id = context.node_id
    centerlized_trainer.logger.info(
        f"[Federated Evaluate] Starting evaluation for client_id={client_id}",
    )

    eval_configs = msg.content.get(
        "config",
        {
            "csv_path": (  # noqa: E501
                r"C:\Users\User\Projects\FYP2\Training_Sample_5pct"
                r"\stage2_train_metadata.csv"
            ),
            "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
        },
    )

    configs = eval_configs
    train_df, val_df = prepare_dataset(
        csv_path=configs["csv_path"],
        image_dir=configs["image_dir"],
        config=config,
        logger=centerlized_trainer.logger,
    )

    train_df = _prepare_evaluation_dataframe(train_df)
    val_df = _prepare_evaluation_dataframe(val_df)

    data_module = create_data_module(
        train_df=train_df,
        val_df=val_df,
        image_dir=configs["image_dir"],
        config=config,
        logger=centerlized_trainer.logger,
    )
    data_module.setup(stage="validate")
    val_loader = data_module.val_dataloader()

    model, callbacks, metrics_collector = _build_model_components(
        centerlized_trainer,
        train_df,
        context,
        is_federated=False,
    )

    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state_dict)

    first_param_name = list(model.state_dict().keys())[0]
    first_param_value = model.state_dict()[first_param_name]
    centerlized_trainer.logger.info(
        f"[Client Evaluate] Loaded model - first param '{first_param_name}' "  # noqa: E501
        f"mean: {first_param_value.mean().item():.6f}",
    )

    trainer = _build_trainer_component(
        centerlized_trainer,
        callbacks,
        is_federated=False,
    )

    results = trainer.test(model, val_loader)
    result_dict = results[0] if results else {}

    centerlized_trainer.logger.info(
        f"[Client Evaluate] Raw result_dict keys: {list(result_dict.keys())}",
    )
    centerlized_trainer.logger.info(f"[Client Evaluate] Raw result_dict: {result_dict}")

    loss, accuracy, precision, recall, f1, auroc = _extract_metrics_from_result(
        result_dict,
    )

    num_examples = len(val_df)

    metric_dict = _create_metric_record_dict(
        loss,
        accuracy,
        precision,
        recall,
        f1,
        auroc,
        num_examples,
    )

    centerlized_trainer.logger.info(
        f"[Client Evaluate] Extracted metrics: loss={loss}, acc={accuracy}, "  # noqa: E501
        f"prec={precision}, rec={recall}, f1={f1}, auroc={auroc}, "
        f"num_examples={num_examples}",
    )

    metric_record = MetricRecord(metric_dict)
    content = RecordDict(
        {
            "metrics": metric_record,
        },
    )
    return Message(content=content, reply_to=msg)
