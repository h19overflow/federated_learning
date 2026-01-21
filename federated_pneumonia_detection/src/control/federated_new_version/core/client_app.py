import random

import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_pneumonia_detection.src.control.dl_model.internals.model.xray_data_module import (
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
    # Initialize trainer and config
    centerlized_trainer, config = _load_trainer_and_config()

    # Extract client_id and round_number from Flower context for federated metrics tracking
    client_id = context.node_id
    round_number = (
        context.state.current_round if hasattr(context.state, "current_round") else 0
    )
    centerlized_trainer.logger.info(
        f"[Federated Train] Starting training for client_id={client_id}, round={round_number}",
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

    # Get seed from server config for reproducible training
    seed = configs.get("seed", 42)

    # Set global seeds for full reproducibility (data splits, augmentation, shuffling)
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
        f"[Federated Train] Using run_id={run_id} from server config",
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

    # Load global model weights from server
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state_dict)

    # Debug: Log model state before training
    first_param_name = list(model.state_dict().keys())[0]
    first_param_before = model.state_dict()[first_param_name].clone()
    centerlized_trainer.logger.info(
        f"[Client Train] BEFORE training - first param '{first_param_name}' mean: {first_param_before.mean().item():.6f}",
    )

    trainer = _build_trainer_component(
        centerlized_trainer,
        callbacks,
        is_federated=True,
    )

    # Train model and collect results
    trainer.fit(model, data_module)

    # Debug: Log model state after training
    first_param_after = model.state_dict()[first_param_name]
    centerlized_trainer.logger.info(
        f"[Client Train] AFTER training - first param '{first_param_name}' mean: {first_param_after.mean().item():.6f}",
    )
    centerlized_trainer.logger.info(
        f"[Client Train] Parameter change: {(first_param_after - first_param_before).abs().mean().item():.6f}",
    )
    results = centerlized_trainer._collect_training_results(
        trainer=trainer,
        model=model,
        metrics_collector=metrics_collector,
    )

    # Number of training examples (CRITICAL for weighted aggregation)
    num_examples = len(train_df)

    # Filter and prepare metrics - IMPORTANT: num-examples at root level for aggregation
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

    # Add num-examples at the root level for Flower's weighted aggregation
    # CRITICAL: Must be "num-examples" with HYPHEN, not underscore!
    metrics_history["num-examples"] = num_examples

    centerlized_trainer.logger.info(
        f"[Client Train] Completed training with {num_examples} examples",
    )

    # Create and return response following Flower conventions
    # CRITICAL: Keys MUST be "arrays" and "metrics" for FedAvg to work
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(metrics_history)
    content = RecordDict(
        {
            "arrays": model_record,  # MUST be "arrays" not "model"
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
    centerlized_trainer, _ = _load_trainer_and_config()

    # Extract client_id for logging
    client_id = context.node_id
    centerlized_trainer.logger.info(
        f"[Federated Evaluate] Starting evaluation for client_id={client_id}",
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
        train_df=train_df,
        val_df=val_df,
        image_dir=configs["image_dir"],
    )
    data_module.setup(stage="validate")
    val_loader = data_module.val_dataloader()

    # Build model and trainer
    model, callbacks, metrics_collector = _build_model_components(
        centerlized_trainer,
        train_df,
        context,
        is_federated=False,
    )

    # Load global model weights
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state_dict)

    # Debug: Check model state to verify it's changing between rounds
    first_param_name = list(model.state_dict().keys())[0]
    first_param_value = model.state_dict()[first_param_name]
    centerlized_trainer.logger.info(
        f"[Client Evaluate] Loaded model - first param '{first_param_name}' mean: {first_param_value.mean().item():.6f}",
    )

    trainer = _build_trainer_component(
        centerlized_trainer,
        callbacks,
        is_federated=False,
    )

    # Evaluate and extract metrics
    results = trainer.test(model, val_loader)
    result_dict = results[0] if results else {}

    # Debug: Print what metrics are actually returned
    centerlized_trainer.logger.info(
        f"[Client Evaluate] Raw result_dict keys: {list(result_dict.keys())}",
    )
    centerlized_trainer.logger.info(f"[Client Evaluate] Raw result_dict: {result_dict}")

    loss, accuracy, precision, recall, f1, auroc = _extract_metrics_from_result(
        result_dict,
    )

    # Number of evaluation examples (CRITICAL for weighted aggregation)
    num_examples = len(val_df)

    # Create metric record - IMPORTANT: num_examples for weighted averaging
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
        f"[Client Evaluate] Extracted metrics: loss={loss}, acc={accuracy}, prec={precision}, rec={recall}, f1={f1}, auroc={auroc}, num_examples={num_examples}",
    )

    metric_record = MetricRecord(metric_dict)
    content = RecordDict(
        {
            "metrics": metric_record,
        },
    )
    return Message(content=content, reply_to=msg)
