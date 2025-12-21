"""
Comparative Analysis Module for Federated vs Centralized Learning.

This module provides tools for running comparative experiments between
federated and centralized learning approaches for pneumonia detection,
with publication-ready statistical analysis and visualization.

Usage:
    python -m analysis.run_analysis --source-path data/Training.zip
"""

from analysis.config import AnalysisConfig

__all__ = ["AnalysisConfig"]
