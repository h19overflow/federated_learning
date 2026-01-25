"""Session management for chat experiences."""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple


from federated_pneumonia_detection.src.boundary.CRUD.chat_history import (
    create_chat_session,
    delete_chat_session,
    get_all_chat_sessions,
    get_chat_session,
)
from federated_pneumonia_detection.src.boundary.models.chat_session import ChatSession
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history import (
    ChatHistoryManager,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.titles import (
    generate_chat_title,
)

logger = logging.getLogger(__name__)


class SessionManager:
    """Coordinates chat session persistence and history cleanup (Singleton)."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        # Only initialize once
        if self._initialized:
            return
        self._history_manager = ChatHistoryManager()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "SessionManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls()
        return cls._instance

    def list_sessions(self) -> List[ChatSession]:
        """Return all chat sessions ordered by last update."""
        return get_all_chat_sessions()

    def create_session(
        self,
        title: Optional[str] = None,
        initial_query: Optional[str] = None,
    ) -> ChatSession:
        """Create a new chat session with optional title generation."""
        if not title and initial_query:
            title = generate_chat_title(initial_query)
            logger.info("[SessionManager] Generated title: '%s'", title)
        return create_chat_session(title=title)

    def ensure_session(self, session_id: str, query: str) -> None:
        """Ensure a database-backed session exists for a session id."""
        try:
            existing = get_chat_session(session_id)
            if not existing:
                create_chat_session(title=f"{query[:50]}...", session_id=session_id)
        except Exception as exc:
            logger.warning(
                "[SessionManager] Failed to ensure DB session (non-fatal): %s",
                exc,
            )

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        return delete_chat_session(session_id)

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._history_manager.clear_history(session_id)

    def get_session_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of (user_message, ai_response) tuples
        """
        return self._history_manager.get_history(session_id)
