"""Chat history management with PostgreSQL persistence."""

from __future__ import annotations

import logging
import uuid
from typing import List, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
from psycopg_pool import ConnectionPool

from federated_pneumonia_detection.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


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
        Initialize the history manager with connection pooling.

        Args:
            table_name: PostgreSQL table name for message storage
            max_history: Maximum conversation turns to keep
        """
        self.table_name = table_name
        self.max_history = max_history
        self._tables_created = False

        # Initialize connection pool (reuse connections instead of creating new ones)
        conn_info = settings.get_postgres_db_uri()
        self._pool = ConnectionPool(
            conninfo=conn_info,
            min_size=2,  # Keep 2 connections ready
            max_size=10,  # Allow up to 10 connections
            timeout=30,  # Wait 30s for available connection
            max_idle=300,  # Close idle connections after 5 minutes
            max_lifetime=3600,  # Recycle connections after 1 hour
        )
        logger.info("[HistoryManager] Initialized connection pool (min=2, max=10)")

    def _get_postgres_history(self, session_id: str) -> PostgresChatMessageHistory:
        """
        Initialize PostgresChatMessageHistory for a session using pooled connection.

        Args:
            session_id: Session identifier (can be any string)

        Returns:
            PostgresChatMessageHistory instance for the session
        """
        # Get connection from pool (reuses existing connections)
        sync_connection = self._pool.getconn()

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

    def _return_connection(self, connection) -> None:
        """
        Return a connection back to the pool.

        Args:
            connection: psycopg connection to return
        """
        try:
            self._pool.putconn(connection)
        except Exception as e:
            logger.warning(f"[HistoryManager] Failed to return connection to pool: {e}")

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
        try:
            history.add_messages(
                [HumanMessage(content=user_message), AIMessage(content=ai_response)],
            )
        finally:
            # Always return connection to pool
            self._return_connection(history._connection)

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of (user_message, ai_response) tuples
        """
        history = self._get_postgres_history(session_id)
        try:
            messages = history.messages

            # Convert to Tuple[str, str] pairs
            formatted_history = []
            for i in range(0, len(messages) - 1, 2):
                if isinstance(messages[i], HumanMessage) and isinstance(
                    messages[i + 1],
                    AIMessage,
                ):
                    formatted_history.append(
                        (messages[i].content, messages[i + 1].content)
                    )
            return formatted_history
        finally:
            # Always return connection to pool
            self._return_connection(history._connection)

    def clear_history(self, session_id: str) -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier
        """
        history = self._get_postgres_history(session_id)
        try:
            history.clear()
        finally:
            # Always return connection to pool
            self._return_connection(history._connection)

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
        try:
            messages = history.messages

            # Enforce limit: keep only most recent N messages
            if limit and len(messages) > limit:
                messages = messages[-limit:]

            return messages
        finally:
            # Always return connection to pool
            self._return_connection(history._connection)

    def close(self) -> None:
        """
        Close the connection pool and release all resources.
        Call this during application shutdown.
        """
        if hasattr(self, "_pool"):
            self._pool.close()
            logger.info("[HistoryManager] Connection pool closed")
