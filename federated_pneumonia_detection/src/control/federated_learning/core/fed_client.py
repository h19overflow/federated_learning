import logging
from flwr.client import NumPyClient
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead
from federated_pneumonia_detection.src.control.federated_learning.training.functions import (
    train_one_epoch,
    evaluate_model,
    get_model_parameters,
    set_model_parameters,
    create_optimizer
)
from typing import List, Dict, Any, Tuple
import torch

class FlowerClient(NumPyClient):
    """
    Flower NumPy client that performs local training on a data partition.
    Each client instance represents one federated learning node.
    """

    def __init__(
        self,
        client_id: int,
        train_loader,
        val_loader,
        constants: SystemConstants,
        config: ExperimentConfig,
        logger: logging.Logger
    ):
        """
        Initialize Flower client with data and configuration.

        Args:
            client_id: Unique identifier for this client
            train_loader: Training DataLoader for this client's partition
            val_loader: Validation DataLoader for this client's partition
            constants: SystemConstants for configuration
            config: ExperimentConfig for training parameters
            logger: Logger instance
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.constants = constants
        self.config = config
        self.logger = logger

        # Create model instance for this client
        self.model = ResNetWithCustomHead(
            constants=constants,
            config=config,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
            fine_tune_layers_count=config.fine_tune_layers_count
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.logger.info(f"Client {client_id} initialized with {len(self.train_loader.dataset)} train samples")

    def get_parameters(self, config: Dict[str, Any]) -> List:
        """Get current model parameters."""
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List) -> None:
        """Set model parameters from server."""
        set_model_parameters(self.model, parameters)

    def fit(self, parameters: List, config: Dict[str, Any]) -> Tuple[List, int, Dict]:
        """
        Train model on local data.

        Args:
            parameters: Model parameters from server
            config: Training configuration from server

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        try:
            self.logger.info(f"[Client {self.client_id}] fit() called - Starting training")
            
            # Set parameters from server
            self.logger.debug(f"[Client {self.client_id}] Setting parameters from server...")
            self.set_parameters(parameters)

            # Get training parameters
            local_epochs = config.get("local_epochs", self.config.local_epochs)
            learning_rate = config.get("lr", self.config.learning_rate)

            # Create optimizer
            optimizer = create_optimizer(self.model, learning_rate, self.config.weight_decay)

            # Training loop
            total_loss = 0.0
            for epoch in range(local_epochs):
                self.logger.debug(f"[Client {self.client_id}] Starting epoch {epoch+1}/{local_epochs}")
                
                epoch_loss = train_one_epoch(
                    model=self.model,
                    dataloader=self.train_loader,
                    optimizer=optimizer,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    logger=self.logger
                )
                total_loss += epoch_loss
                self.logger.debug(f"[Client {self.client_id}] Epoch {epoch+1} loss: {epoch_loss:.4f}")

            avg_loss = total_loss / local_epochs
            self.logger.info(f"[Client {self.client_id}] Training complete - avg_loss={avg_loss:.4f}")

            # Get updated parameters
            self.logger.debug(f"[Client {self.client_id}] Extracting updated parameters...")
            updated_parameters = self.get_parameters(config={})
            self.logger.debug(f"[Client {self.client_id}] Extracted {len(updated_parameters)} parameter arrays")

            # Return results
            num_examples = len(self.train_loader.dataset)
            metrics = {
                "train_loss": avg_loss,
                "client_id": self.client_id
            }

            self.logger.info(
                f"[Client {self.client_id}] fit() completed successfully: "
                f"loss={avg_loss:.4f}, samples={num_examples}"
            )

            return updated_parameters, num_examples, metrics
            
        except Exception as e:
            self.logger.error(f"[Client {self.client_id}] CRITICAL ERROR in fit(): {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"[Client {self.client_id}] Traceback:\n{traceback.format_exc()}")
            raise

    def evaluate(self, parameters: List, config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local validation data.

        Args:
            parameters: Model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        try:
            self.logger.info(f"[Client {self.client_id}] evaluate() called - Starting evaluation")
            
            # Set parameters from server
            self.logger.debug(f"[Client {self.client_id}] Setting parameters from server...")
            self.set_parameters(parameters)

            # Evaluate
            loss, accuracy, metrics = evaluate_model(
                model=self.model,
                dataloader=self.val_loader,
                device=self.device,
                num_classes=self.config.num_classes,
                logger=self.logger
            )

            num_examples = len(self.val_loader.dataset)
            metrics_out = {
                "accuracy": accuracy,
                "client_id": self.client_id
            }

            self.logger.info(
                f"[Client {self.client_id}] evaluate() completed successfully: "
                f"loss={loss:.4f}, acc={accuracy:.4f}, samples={num_examples}"
            )

            return loss, num_examples, metrics_out
            
        except Exception as e:
            self.logger.error(f"[Client {self.client_id}] CRITICAL ERROR in evaluate(): {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"[Client {self.client_id}] Traceback:\n{traceback.format_exc()}")
            raise
