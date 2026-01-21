"""Shared contracts for multi-agent chat systems."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, TypedDict


class AgentEventType(str, Enum):
    """Supported event types for agent streaming."""

    SESSION = "session"
    TOKEN = "token"
    STATUS = "status"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    DONE = "done"


@dataclass(frozen=True)
class ChatInput:
    """Input payload for chat streaming and query execution."""

    query: str
    session_id: str
    arxiv_enabled: bool = False
    run_id: Optional[int] = None
    original_query: Optional[str] = None


class AgentEvent(TypedDict, total=False):
    """Streaming event emitted by chat agents."""

    type: str
    content: str
    message: str
    tool: str
    args: Dict[str, Any]
    session_id: str
