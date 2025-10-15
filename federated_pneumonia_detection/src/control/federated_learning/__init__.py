"""
Federated learning system for pneumonia detection.
Organized by functional responsibility for intuitive code navigation.

Directory Structure:
    core/           - Flower infrastructure (SimulationRunner)
    data/           - Data management (partitioning, client DataLoaders)
    training/       - Training utilities (pure PyTorch functions)
    federated_trainer.py - Main entry point (orchestrator)

Usage:
    from federated_pneumonia_detection.src.control.federated_learning import FederatedTrainer

    trainer = FederatedTrainer(
        partition_strategy='iid',  # or 'non-iid', 'stratified'
        checkpoint_dir='federated_checkpoints'
    )

    results = trainer.train(
        source_path='path/to/data.zip',
        experiment_name='my_fl_experiment'
    )
"""

from .federated_trainer import FederatedTrainer

__all__ = ['FederatedTrainer']
