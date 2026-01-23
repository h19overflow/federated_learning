"""Chat history management with PostgreSQL persistence."""

from __future__ import annotations

import logging
import uuid
from typing import List, Tuple

import psycopg
from langchain_core.messages import AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory

from federated_pneumonia_detection.src.boundary.engine import settings

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """
    Manages chat history sessions with PostgreSQL persistence.

    Uses langchain-postgres's PostgresChatMessageHistory for storage,
    with automatic UUID conversion for session IDs.
    """

    def __init__(
        self,
        table_name: str = "message_store",
        max_history: int = 10,
    ) -> None:
        """
        Initialize the history manager.

        Args:
            table_name: PostgreSQL table name for message storage
            max_history: Maximum conversation turns to keep (not currently enforced)
        """
        self.table_name = table_name
        self.max_history = max_history
        self._tables_created = False

    def _get_postgres_history(self, session_id: str) -> PostgresChatMessageHistory:
        """
        Initialize PostgresChatMessageHistory for a session.

        Args:
            session_id: Session identifier (can be any string)

        Returns:
            PostgresChatMessageHistory instance for the session
        """
        conn_info = settings.get_postgres_db_uri()
        sync_connection = psycopg.connect(conn_info)

        # Validate or convert session_id to UUID
        try:
            uuid.UUID(session_id)
            clean_session_id = session_id
        except ValueError:
            # Generate deterministic UUID from string using UUID5
            clean_session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id))
            logger.info(
                f"[HistoryManager] Mapped session_id '{session_id}' to UUID '{clean_session_id}'",
            )

        history = PostgresChatMessageHistory(
            self.table_name,
            clean_session_id,
            sync_connection=sync_connection,
        )
        # Ensure tables exist (only on first use)
        if not self._tables_created:
            history.create_tables(sync_connection, self.table_name)
            self._tables_created = True
        return history

    def add_to_history(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
    ) -> None:
        """
        Add a conversation turn to session history.

        Args:
            session_id: Unique session identifier
            user_message: User's query
            ai_response: AI's response
        """
        history = self._get_postgres_history(session_id)
        history.add_messages(
            [HumanMessage(content=user_message), AIMessage(content=ai_response)],
        )

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of (user_message, ai_response) tuples
        """
        history = self._get_postgres_history(session_id)
        messages = history.messages

        # Convert to Tuple[str, str] pairs
        formatted_history = []
        for i in range(0, len(messages) - 1, 2):
            if isinstance(messages[i], HumanMessage) and isinstance(
                messages[i + 1],
                AIMessage,
            ):
                formatted_history.append((messages[i].content, messages[i + 1].content))
        return formatted_history

    def clear_history(self, session_id: str) -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier
        """
        history = self._get_postgres_history(session_id)
        history.clear()

    def format_for_context(self, session_id: str) -> str:
        """
        Format conversation history as a context string.

        Args:
            session_id: Session identifier

        Returns:
            Formatted history string for inclusion in prompts
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        formatted = ""
        for user_msg, ai_msg in history:
            formatted += f"User: {user_msg}\nAssistant: {ai_msg}\n\n"
        return formatted.strip()

    def get_messages(self, session_id: str, limit: int | None = None) -> list:
        """
        Get raw LangChain messages for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return (most recent first)

        Returns:
            List of LangChain message objects
        """
        history = self._get_postgres_history(session_id)
        messages = history.messages

        # Enforce limit: keep only most recent N messages
        if limit and len(messages) > limit:
            messages = messages[-limit:]

        return messages
