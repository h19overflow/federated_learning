"""

Custom Flower strategy that sends training and evaluation configurations to clients.
"""

from typing import Iterable, Optional, Dict, Any
from flwr.app import ArrayRecord, ConfigRecord, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class ConfigurableFedAvg(FedAvg):
    """
    FedAvg strategy extended to include custom configurations in train/evaluate messages.
    Ensures clients receive necessary configuration parameters for training and evaluation.
    """

    def __init__(
        self,
        train_config: Optional[Dict[str, Any]] = None,
        eval_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize ConfigurableFedAvg strategy.

        Args:
            train_config: Dictionary of configuration parameters for training
            eval_config: Dictionary of configuration parameters for evaluation
            **kwargs: Additional arguments passed to FedAvg
        """
        super().__init__(**kwargs)
        self.train_config = train_config or {}
        self.eval_config = eval_config or {}

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training with custom configs.
        
        Args:
            server_round: Current round of federated learning
            arrays: Current global ArrayRecord (model) to send to clients
            config: Configuration to be sent to clients for training
            grid: Grid instance for node sampling and communication

        Returns:
            Iterable of messages to be sent to selected client nodes for training
        """
        # Merge custom train config into the base config
        config.update(self.train_config)
        # Call parent class to configure training with updated config
        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation with custom configs.
        
        Args:
            server_round: Current round of federated learning
            arrays: Current global ArrayRecord (model) to send to clients
            config: Configuration to be sent to clients for evaluation
            grid: Grid instance for node sampling and communication

        Returns:
            Iterable of messages to be sent to selected client nodes for evaluation
        """
        # Merge custom eval config into the base config
        config.update(self.eval_config)
        # Call parent class to configure evaluation with updated config
        return super().configure_evaluate(server_round, arrays, config, grid)
