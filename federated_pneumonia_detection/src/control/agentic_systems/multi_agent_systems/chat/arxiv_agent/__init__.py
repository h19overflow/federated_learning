"""Arxiv Agent package.

Provides research assistance using local RAG and arxiv search capabilities.
"""

from .engine import ArxivAugmentedEngine
from .streaming import SSEEventType, create_sse_event

__all__ = [
    "ArxivAugmentedEngine",
    "SSEEventType",
    "create_sse_event",
]
