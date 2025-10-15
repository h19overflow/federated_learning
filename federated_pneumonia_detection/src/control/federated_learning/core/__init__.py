"""
Core federated learning infrastructure.
Handles Flower client/server apps and simulation orchestration.
"""

from .simulation_runner import SimulationRunner

__all__ = ['SimulationRunner']
