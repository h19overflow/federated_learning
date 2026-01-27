"""
Configuration module for Federated Pneumonia Detection System

This module provides centralized configuration management through the
ConfigManager class.
"""

from .config_manager import ConfigManager, get_config_manager, quick_get, quick_set

__all__ = ["ConfigManager", "get_config_manager", "quick_get", "quick_set"]
