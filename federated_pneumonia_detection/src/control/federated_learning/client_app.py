"""
Flower federated learning client application.
Integrates ResNetWithCustomHead model with federated learning workflow using existing utilities.
"""

import logging
import torch
from flwr.client import ClientApp
from flwr.common import Context, Message, ArrayRecord, MetricRecord, RecordDict

from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead
from federated_pneumonia_detection.src.control.federated_learning.training_functions import (
    train_one_epoch,
    evaluate_model,
    create_optimizer,
    set_model_parameters,
    get_model_parameters
)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flower client app
app = ClientApp()


def load_client_config():
    """Load configuration from YAML using ConfigLoader."""
    config_loader = ConfigLoader(config_dir="federated_pneumonia_detection/config")
    constants = config_loader.create_system_constants()
    config = config_loader.create_experiment_config()
    return constants, config


def create_client_model(constants, config):
    """Create ResNetWithCustomHead model instance."""
    model = ResNetWithCustomHead(
        constants=constants,
        config=config,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        fine_tune_layers_count=config.fine_tune_layers_count
    )
    return model


@app.train()
def train(msg: Message, context: Context):
    """
    Train the model on local client data.

    Args:
        msg: Message containing server model weights and config
        context: Flower context with node and run configuration

    Returns:
        Message with updated weights and training metrics
    """
    logger.info("Starting client training...")

    # Load configuration
    constants, config = load_client_config()

    # Create model
    model = create_client_model(constants, config)

    # Load weights from server
    server_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(server_state_dict)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get training parameters from context
    local_epochs = context.run_config.get("local-epochs", config.local_epochs)
    learning_rate = msg.content.get("config", {}).get("lr", config.learning_rate)

    # Note: In actual implementation, you would load the client's data partition here
    # This requires the FederatedTrainer to set up data partitions and pass partition_id
    # For now, this is a placeholder - data loading will be handled by FederatedTrainer
    partition_id = context.node_config.get("partition-id", 0)
    logger.info(f"Training on partition {partition_id} for {local_epochs} epochs")

    # Create optimizer
    optimizer = create_optimizer(model, learning_rate, config.weight_decay)

    # Training loop (simplified - actual data loading will be in FederatedTrainer)
    # In production, you'd get the trainloader from context or load data here
    # For now, we return a placeholder structure

    # Placeholder for training (will be properly implemented with data loaders)
    logger.warning(
        "Client training requires data loaders to be set up by FederatedTrainer. "
        "This is a template implementation."
    )

    # Construct response
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": 0.0,  # Placeholder
        "num_examples": 0,   # Placeholder
        "partition_id": partition_id
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    logger.info("Client training completed")
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """
    Evaluate the model on local validation data.

    Args:
        msg: Message containing server model weights
        context: Flower context with node configuration

    Returns:
        Message with evaluation metrics
    """
    logger.info("Starting client evaluation...")

    # Load configuration
    constants, config = load_client_config()

    # Create model
    model = create_client_model(constants, config)

    # Load weights from server
    server_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(server_state_dict)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get partition ID
    partition_id = context.node_config.get("partition-id", 0)
    logger.info(f"Evaluating on partition {partition_id}")

    # Placeholder for evaluation (will be properly implemented with data loaders)
    logger.warning(
        "Client evaluation requires data loaders to be set up by FederatedTrainer. "
        "This is a template implementation."
    )

    # Construct response
    metrics = {
        "eval_loss": 0.0,   # Placeholder
        "eval_acc": 0.0,    # Placeholder
        "num_examples": 0,  # Placeholder
        "partition_id": partition_id
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    logger.info("Client evaluation completed")
    return Message(content=content, reply_to=msg)