"""
Backward compatibility re-export for arxiv_agent module.

The ArxivAugmentedEngine has been refactored into the arxiv_agent package.
This module re-exports it to maintain backward compatibility with existing imports.

Import example:
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent import (
        ArxivAugmentedEngine,
    )

New preferred import (directly from package):
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent import (
        ArxivAugmentedEngine,
        SSEEventType,
        create_sse_event,
    )
"""

# Re-export from the refactored package
# Note: Since this file is arxiv_agent.py and the package is arxiv_agent/,
# Python will prefer the .py file. We need to import from the package submodules directly.
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent.engine import (
    ArxivAugmentedEngine,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent.streaming import (
    SSEEventType,
    create_sse_event,
)

__all__ = [
    "ArxivAugmentedEngine",
    "SSEEventType",
    "create_sse_event",
]
