"""
Flower federated learning server application.
Coordinates federated training using ResNetWithCustomHead and configuration from YAML.
"""

import logging
import os
import torch
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from flwr.common import Context, ArrayRecord, ConfigRecord

from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create ServerApp instance
app = ServerApp()


def load_server_config():
    """Load configuration from YAML using ConfigLoader."""
    config_loader = ConfigLoader(config_dir="federated_pneumonia_detection/config")
    constants = config_loader.create_system_constants()
    config = config_loader.create_experiment_config()
    return constants, config


def create_global_model(constants, config):
    """Create global ResNetWithCustomHead model instance."""
    logger.info("Initializing global model...")

    model = ResNetWithCustomHead(
        constants=constants,
        config=config,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        fine_tune_layers_count=config.fine_tune_layers_count
    )

    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Global model created: {model_info['total_parameters']} parameters")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']}")

    return model


@app.main()
def main(grid, context: Context) -> None:
    """
    Main entry point for the federated learning server.

    Args:
        grid: Flower grid for client coordination
        context: Context containing run configuration
    """
    logger.info("Starting Federated Learning Server...")

    # Load configuration from YAML
    constants, config = load_server_config()

    # Read run config (with defaults from YAML config)
    fraction_fit = context.run_config.get("fraction-fit", float(config.clients_per_round) / config.num_clients)
    num_rounds = context.run_config.get("num-server-rounds", config.num_rounds)
    learning_rate = context.run_config.get("lr", config.learning_rate)

    logger.info(f"Federated Learning Configuration:")
    logger.info(f"  - Number of rounds: {num_rounds}")
    logger.info(f"  - Fraction fit: {fraction_fit}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Total clients: {config.num_clients}")
    logger.info(f"  - Clients per round: {config.clients_per_round}")

    # Create global model
    global_model = create_global_model(constants, config)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    logger.info("Initializing FedAvg strategy...")
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,  # Evaluate on all clients
        min_fit_clients=config.clients_per_round,
        min_available_clients=config.num_clients
    )

    # Start federated learning
    logger.info(f"Starting {num_rounds} rounds of federated learning...")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": learning_rate}),
        num_rounds=num_rounds,
    )

    # Save final model to checkpoint directory
    checkpoint_dir = getattr(config, 'checkpoint_dir', 'models/checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    final_model_path = os.path.join(checkpoint_dir, "federated_final_model.pt")

    logger.info(f"Saving final model to {final_model_path}...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, final_model_path)

    logger.info("Federated learning completed successfully!")
    logger.info(f"Final model saved to: {final_model_path}")
