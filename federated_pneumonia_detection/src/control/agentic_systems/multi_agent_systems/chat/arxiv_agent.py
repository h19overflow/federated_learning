"""
Arxiv Augmented Engine - LangChain agent with arxiv and local RAG tools.

Provides research assistance using both local knowledge base and arxiv search.

Dependencies:
    - langchain.agents.create_agent
    - langchain_google_genai.ChatGoogleGenerativeAI
    - MCPManager for arxiv tools
    - RAG tool for local search
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent_prompts import (
    ARXIV_AGENT_SYSTEM_PROMPT,
    format_user_prompt,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.mcp_manager import (
    MCPManager,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.tools.rag_tool import (
    create_rag_tool,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.retriver import (
    QueryEngine,
)

logger = logging.getLogger(__name__)


class ArxivAugmentedEngine:
    """
    Research agent combining local RAG and arxiv search capabilities.

    Uses LangChain agent with tool-calling for intelligent search decisions.
    """

    def __init__(self, max_history: int = 10) -> None:
        """
        Initialize the arxiv-augmented engine.

        Args:
            max_history: Maximum conversation turns to keep in memory
        """
        self.max_history = max_history
        self.conversation_history: Dict[str, List[Tuple[str, str]]] = {}

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.7,
        )

        # Initialize RAG tool with QueryEngine (optional - may fail if DB unavailable)
        self._query_engine = None
        self._rag_tool = None
        try:
            self._query_engine = QueryEngine()
            self._rag_tool = create_rag_tool(self._query_engine)
            logger.info("RAG tool initialized successfully")
        except Exception as e:
            logger.warning(f"RAG tool unavailable (database not connected): {e}")
            logger.info("ArxivAugmentedEngine will work with arxiv tools only")

    def _get_tools(self, arxiv_enabled: bool) -> list:
        """
        Build tool list based on arxiv availability.

        Args:
            arxiv_enabled: Whether to include arxiv tools

        Returns:
            List of LangChain tools
        """
        tools = []

        # Always add RAG tool if available
        if self._rag_tool is not None:
            tools.append(self._rag_tool)

        # Add arxiv tools if enabled and available
        if arxiv_enabled:
            mcp_manager = MCPManager.get_instance()
            if mcp_manager.is_available:
                arxiv_tools = mcp_manager.get_arxiv_tools()
                tools.extend(arxiv_tools)
                logger.info(f"Added {len(arxiv_tools)} arxiv tools")
            else:
                logger.warning("Arxiv requested but MCP manager not available")

        return tools

    def add_to_history(
        self, session_id: str, user_message: str, ai_response: str
    ) -> None:
        """
        Add a conversation turn to session history.

        Args:
            session_id: Unique session identifier
            user_message: User's query
            ai_response: AI's response
        """
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        self.conversation_history[session_id].append((user_message, ai_response))

        # Trim to max history
        if len(self.conversation_history[session_id]) > self.max_history:
            self.conversation_history[session_id] = self.conversation_history[
                session_id
            ][-self.max_history :]

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Get conversation history for a session."""
        return self.conversation_history.get(session_id, [])

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]

    def _format_history_for_context(self, session_id: str) -> str:
        """Format conversation history as context string."""
        history = self.get_history(session_id)
        if not history:
            return ""

        formatted = ""
        for user_msg, ai_msg in history:
            formatted += f"User: {user_msg}\nAssistant: {ai_msg}\n\n"
        return formatted.strip()

    def _build_messages(
        self, query: str, session_id: str
    ) -> List[SystemMessage | HumanMessage | AIMessage]:
        """
        Build message list for the agent.

        Args:
            query: Current user query
            session_id: Session identifier for history

        Returns:
            List of messages for the agent
        """
        messages = [SystemMessage(content=ARXIV_AGENT_SYSTEM_PROMPT)]

        # Add history as alternating messages
        history = self.get_history(session_id)
        for user_msg, ai_msg in history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=ai_msg))

        # Add current query
        messages.append(HumanMessage(content=query))

        return messages

    async def query_stream(
        self, query: str, session_id: str, arxiv_enabled: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream query response token by token.

        Args:
            query: User's question
            session_id: Session identifier for conversation tracking
            arxiv_enabled: Whether to enable arxiv search tools

        Yields:
            Dict with type and content for each streamed chunk
        """
        tools = self._get_tools(arxiv_enabled)

        if not tools:
            yield {"type": "error", "message": "No tools available"}
            return

        try:
            # Bind tools to the model
            model_with_tools = self.llm.bind_tools(tools)

            # Build messages with history
            messages = self._build_messages(query, session_id)

            full_response = ""
            tool_calls_made = []

            # First call - check if model wants to use tools
            response = await model_with_tools.ainvoke(messages)

            # Check for tool calls
            if response.tool_calls:
                yield {"type": "status", "content": "Searching..."}

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": tool_args,
                    }

                    # Execute tool
                    tool_result = None
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = tool.invoke(tool_args)
                            except Exception as e:
                                tool_result = f"Tool error: {str(e)}"
                            break

                    tool_calls_made.append(
                        {
                            "name": tool_name,
                            "args": tool_args,
                            "result": tool_result,
                        }
                    )

                # Build follow-up messages with tool results
                from langchain_core.messages import ToolMessage

                messages.append(response)
                for tc, made in zip(response.tool_calls, tool_calls_made):
                    messages.append(
                        ToolMessage(
                            content=str(made["result"]),
                            tool_call_id=tc["id"],
                        )
                    )

                # Stream final response
                async for chunk in model_with_tools.astream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        full_response += chunk.content
                        yield {"type": "token", "content": chunk.content}
            else:
                # No tools needed, stream direct response
                async for chunk in model_with_tools.astream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        full_response += chunk.content
                        yield {"type": "token", "content": chunk.content}

            # Save to history
            self.add_to_history(session_id, query, full_response)
            yield {"type": "done", "session_id": session_id}

        except Exception as e:
            logger.error(f"Error in query_stream: {e}")
            yield {"type": "error", "message": str(e)}

    async def query(
        self, query: str, session_id: str, arxiv_enabled: bool = False
    ) -> Dict[str, Any]:
        """
        Non-streaming query (collects full response).

        Args:
            query: User's question
            session_id: Session identifier
            arxiv_enabled: Whether to enable arxiv tools

        Returns:
            Dict with answer and metadata
        """
        full_response = ""
        tool_calls = []

        async for chunk in self.query_stream(query, session_id, arxiv_enabled):
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
