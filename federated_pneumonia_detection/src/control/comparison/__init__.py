"""
Comparison module for federated pneumonia detection system.
Provides orchestration for comparing centralized and federated learning approaches.
"""

from .experiment_orchestrator import ExperimentOrchestrator, run_quick_comparison

__all__ = [
    'ExperimentOrchestrator',
    'run_quick_comparison'
]
