"""Shared streaming state for research agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StreamState:
    """Accumulates streaming output for persistence."""

    full_response: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
