"""
Federated learning strategy configuration.

This module provides factory functions to create and configure Flower
aggregation strategies for federated learning.

Dependencies:
- flwr.server.strategy: Built-in Flower strategies

Role in System:
- Configures server-side aggregation strategy (FedAvg)
- Defines client sampling and parameter aggregation rules
- Injected by FederatedTrainer during simulation initialization
"""

from typing import List

import numpy as np
from flwr.server.strategy import FedAvg


def create_federated_strategy(
    num_clients: int,
    clients_per_round: int,
    initial_parameters: List[np.ndarray]
) -> FedAvg:
    """
    Create and configure FedAvg strategy for federated learning.

    Calculates sampling fractions based on clients available per round
    and configures aggregation strategy with provided initial parameters.

    Args:
        num_clients: Total number of clients in federated learning setup
        clients_per_round: Number of clients to sample per aggregation round
        initial_parameters: Initial model parameters as list of numpy arrays

    Returns:
        Configured FedAvg strategy instance ready for simulation

    Raises:
        ValueError: If num_clients <= 0, clients_per_round <= 0, or
                   clients_per_round > num_clients
    """
    # Validate inputs
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")

    if clients_per_round <= 0:
        raise ValueError(f"clients_per_round must be positive, got {clients_per_round}")

    if clients_per_round > num_clients:
        raise ValueError(
            f"clients_per_round ({clients_per_round}) cannot exceed "
            f"num_clients ({num_clients})"
        )

    if not initial_parameters:
        raise ValueError("initial_parameters cannot be empty")

    # Calculate sampling fractions
    fraction = clients_per_round / num_clients

    # Create and return strategy
    strategy = FedAvg(
        fraction_fit=fraction,
        fraction_evaluate=fraction,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=clients_per_round,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
    )

    return strategy
