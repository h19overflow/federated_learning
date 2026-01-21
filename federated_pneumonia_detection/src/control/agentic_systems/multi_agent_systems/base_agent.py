"""Base interface for chat-capable agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Tuple

from .contracts import AgentEvent, ChatInput


class BaseAgent(ABC):
    """Abstract interface for streaming chat agents."""

    @abstractmethod
    async def stream(self, chat_input: ChatInput) -> AsyncGenerator[AgentEvent, None]:
        """Stream a response for the provided chat input."""

    @abstractmethod
    async def query(self, chat_input: ChatInput) -> Dict[str, Any]:
        """Run a non-streaming query for the provided chat input."""

    @abstractmethod
    def history(self, session_id: str) -> List[Tuple[str, str]]:
        """Return conversation history for a session."""

    @abstractmethod
    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
