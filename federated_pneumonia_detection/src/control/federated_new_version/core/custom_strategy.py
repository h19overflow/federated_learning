"""
Custom Flower strategy that sends training and evaluation configurations to clients.
"""

from typing import Optional, Dict, Any
from flwr.serverapp.strategy import FedAvg
from flwr.app import Message, ConfigRecord


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

    def configure_train_ins(
        self, parameters, config
    ):
        """
        Configure training instructions with custom configs.
        
        Args:
            parameters: Model parameters to send to clients
            config: Run configuration

        Returns:
            List of (client_id, FitIns) tuples with embedded configs
        """
        # Get the default training instructions from parent class
        fit_ins_list = super().configure_train_ins(parameters, config)
        
        # Add configs to each client's training instruction
        if fit_ins_list:
            for i, (client_id, fit_ins) in enumerate(fit_ins_list):
                # Embed configs in the message
                fit_ins.config["configs"] = self.train_config
        
        return fit_ins_list

    def configure_evaluate_ins(
        self, parameters, config
    ):
        """
        Configure evaluation instructions with custom configs.
        
        Args:
            parameters: Model parameters to send to clients
            config: Run configuration

        Returns:
            List of (client_id, EvaluateIns) tuples with embedded configs
        """
        # Get the default evaluation instructions from parent class
        evaluate_ins_list = super().configure_evaluate_ins(parameters, config)
        
        # Add configs to each client's evaluation instruction
        if evaluate_ins_list:
            for i, (client_id, eval_ins) in enumerate(evaluate_ins_list):
                # Embed configs in the message
                eval_ins.config["configs"] = self.eval_config
        
        return evaluate_ins_list
