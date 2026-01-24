"""Arxiv-augmented research engine for chat workflows."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Tuple, cast

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.base_agent import (
    BaseAgent,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.agents.research_stream import (
    stream_query,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.contracts import (
    AgentEvent,
    ChatInput,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history import (
    ChatHistoryManager,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.rag import (
    QueryEngine,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.tools.rag_tool import (
    create_rag_tool,
)

load_dotenv()
logger = logging.getLogger(__name__)


class ArxivAugmentedEngine(BaseAgent):
    """Research agent combining local RAG and arxiv search capabilities."""

    def __init__(self, max_history: int = 10) -> None:
        """
        Initialize the arxiv-augmented engine.

        Args:
            max_history: Maximum conversation turns to keep in memory
        """
        logger.info(f"[ArxivEngine] Initializing with max_history={max_history}")

        # Initialize history manager
        self._history_manager = ChatHistoryManager(max_history=max_history)

        try:
            logger.info("[ArxivEngine] Initializing ChatGoogleGenerativeAI...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=1.0,  # Gemini 3 default - prevents infinite loops
                max_tokens=2048,  # Limit response length for concise answers
            )
            logger.info("[ArxivEngine] LLM initialized successfully")
        except Exception as e:
            logger.error(f"[ArxivEngine] Failed to initialize LLM: {e}", exc_info=True)
            raise

        # Initialize RAG tool (optional - may fail if DB unavailable)
        self._query_engine = None
        self._rag_tool = None
        try:
            logger.info("[ArxivEngine] Initializing RAG tool via QueryEngine...")
            self._query_engine = QueryEngine()
            self._rag_tool = create_rag_tool(self._query_engine)
            logger.info("[ArxivEngine] RAG tool initialized successfully")
        except Exception as e:
            logger.warning(f"[ArxivEngine] RAG tool unavailable: {e}")
            logger.info("[ArxivEngine] Will work with arxiv tools only")

    # =========================================================================
    # History delegation methods
    # =========================================================================

    def add_to_history(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
    ) -> None:
        """Add a conversation turn to session history."""
        self._history_manager.add_to_history(session_id, user_message, ai_response)

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Get conversation history for a session."""
        return self._history_manager.get_history(session_id)

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._history_manager.clear_history(session_id)

    async def query_stream(
        self,
        query: str,
        session_id: str,
        arxiv_enabled: bool = False,
        original_query: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream query response token by token."""
        logger.info(
            "[ArxivEngine] Starting stream for session %s, query: '%s...'",
            session_id,
            query[:50],
        )
        async for event in stream_query(
            self.llm,
            self._history_manager,
            self._rag_tool,
            query,
            session_id,
            arxiv_enabled=arxiv_enabled,
            original_query=original_query,
        ):
            yield event

    # =========================================================================
    # Non-streaming query (internal method)
    # =========================================================================
    async def _query_internal(
        self,
        query: str,
        session_id: str,
        arxiv_enabled: bool = False,
        original_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Non-streaming query (collects full response).

        Args:
            query: User's question
            session_id: Session identifier
            arxiv_enabled: Whether to enable arxiv tools
            original_query: Original query before context enhancement

        Returns:
            Dict with answer and metadata
        """
        full_response = ""
        tool_calls = []

        async for chunk in self.query_stream(
            query,
            session_id,
            arxiv_enabled,
            original_query,
        ):
            if chunk["type"] == "token":
                full_response += chunk["content"]
            elif chunk["type"] == "tool_call":
                tool_calls.append(chunk)
            elif chunk["type"] == "error":
                return {"error": chunk["message"]}

        return {
            "answer": full_response,
            "session_id": session_id,
            "tools_used": tool_calls,
        }

    # =========================================================================
    # BaseAgent contract methods
    # =========================================================================
    async def stream(self, chat_input: ChatInput) -> AsyncGenerator[AgentEvent, None]:
        """Stream agent events for the given chat input (BaseAgent contract)."""
        async for event in self.query_stream(
            chat_input.query,
            chat_input.session_id,
            arxiv_enabled=chat_input.arxiv_enabled,
            original_query=chat_input.original_query,
        ):
            yield cast(AgentEvent, event)

    async def query(self, chat_input: ChatInput) -> Dict[str, Any]:
        """Run a non-streaming query (BaseAgent contract)."""
        return await self._query_internal(
            chat_input.query,
            chat_input.session_id,
            arxiv_enabled=chat_input.arxiv_enabled,
            original_query=chat_input.original_query,
        )

    def history(self, session_id: str) -> List[Tuple[str, str]]:
        """Return conversation history (BaseAgent contract)."""
        return self.get_history(session_id)
