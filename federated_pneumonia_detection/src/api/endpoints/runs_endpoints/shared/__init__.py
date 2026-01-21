"""Shared utilities for runs endpoints.

This module exports shared components used across runs endpoint implementations.
Follows SRP by centralizing common utilities and preventing duplication.
"""

from federated_pneumonia_detection.src.api.endpoints.runs_endpoints.shared.summary_builder import (
    FederatedRunSummarizer,
    RunSummaryBuilder,
)

__all__ = ["RunSummaryBuilder", "FederatedRunSummarizer"]
