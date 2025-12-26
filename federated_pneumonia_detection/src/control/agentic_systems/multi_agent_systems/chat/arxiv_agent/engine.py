"""
Arxiv Augmented Engine - LangChain agent with arxiv and local RAG tools.

Provides research assistance using both local knowledge base and arxiv search.

Dependencies:
    - langchain_google_genai.ChatGoogleGenerativeAI
    - MCPManager for arxiv tools
    - RAG tool for local search
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent_prompts import (
    ARXIV_AGENT_SYSTEM_PROMPT,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.mcp_manager import (
    MCPManager,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.retriver import (
    QueryEngine,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.tools.rag_tool import (
    create_rag_tool,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.tools.arxiv_embedding_tool import (
    create_arxiv_embedding_tool,
)

from .content import chunk_content, normalize_content
from .history import ChatHistoryManager
from .streaming import SSEEventType, create_sse_event, execute_tool_async


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
        logger.info(f"[ArxivEngine] Initializing with max_history={max_history}")
        
        # Initialize history manager
        self._history_manager = ChatHistoryManager(max_history=max_history)

        try:
            logger.info("[ArxivEngine] Initializing ChatGoogleGenerativeAI...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                temperature=0.7,
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
        self, session_id: str, user_message: str, ai_response: str
    ) -> None:
        """Add a conversation turn to session history."""
        self._history_manager.add_to_history(session_id, user_message, ai_response)

    def get_history(self, session_id: str) -> List[tuple]:
        """Get conversation history for a session."""
        return self._history_manager.get_history(session_id)

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._history_manager.clear_history(session_id)

    # =========================================================================
    # Tool management
    # =========================================================================
    
    def _get_tools(self, arxiv_enabled: bool) -> list:
        """
        Build tool list based on arxiv availability.

        Args:
            arxiv_enabled: Whether to include arxiv tools

        Returns:
            List of LangChain tools
        """
        tools = []

        if self._rag_tool is not None:
            tools.append(self._rag_tool)

        if arxiv_enabled:
            mcp_manager = MCPManager.get_instance()
            if mcp_manager.is_available:
                arxiv_tools = mcp_manager.get_arxiv_tools()
                tools.extend(arxiv_tools)
                logger.info(f"Added {len(arxiv_tools)} arxiv tools")
                
                # Add embedding tool for knowledge base expansion
                embedding_tool = create_arxiv_embedding_tool()
                tools.append(embedding_tool)
                logger.info("Added arxiv embedding tool")
            else:
                logger.warning("Arxiv requested but MCP manager not available")

        return tools

    # =========================================================================
    # Message building
    # =========================================================================
    
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
        
        # Add history from persistent store
        messages.extend(self._history_manager.get_messages(session_id))
        
        # Add current query
        messages.append(HumanMessage(content=query))

        return messages

    # =========================================================================
    # Streaming query
    # =========================================================================
    
    async def query_stream(
        self,
        query: str,
        session_id: str,
        arxiv_enabled: bool = False,
        original_query: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream query response token by token.

        Args:
            query: User's question
            session_id: Session identifier for conversation tracking
            arxiv_enabled: Whether to enable arxiv search tools
            original_query: Original query before context enhancement (for history)

        Yields:
            Dict with type and content for each streamed chunk
        """
        logger.info(
            f"[ArxivEngine] Starting stream for session {session_id}, "
            f"query: '{query[:50]}...', arxiv={arxiv_enabled}"
        )

        tools = self._get_tools(arxiv_enabled)
        logger.info(f"[ArxivEngine] Retrieved {len(tools)} tools: {[t.name for t in tools]}")

        if not tools:
            logger.error("[ArxivEngine] No tools available")
            yield create_sse_event(SSEEventType.ERROR, message="No tools available")
            return

        try:
            # Bind tools to model
            model_with_tools = self.llm.bind_tools(tools)
            messages = self._build_messages(query, session_id)
            logger.info(f"[ArxivEngine] Built {len(messages)} messages including history")

            full_response = ""
            tool_calls_made = []

            yield create_sse_event(SSEEventType.STATUS, content="Analyzing your query...")

            # First call - check if model wants to use tools
            logger.info("[ArxivEngine] Invoking model to check for tool calls...")
            response = await model_with_tools.ainvoke(messages)

            # Tool execution loop
            if response.tool_calls:
                while response.tool_calls:
                    logger.info(f"[ArxivEngine] Model requested {len(response.tool_calls)} tool calls")

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        logger.info(f"[ArxivEngine] Executing tool: {tool_name}")

                        # Yield user-friendly status
                        friendly_name = tool_name.replace("_", " ").title()
                        yield create_sse_event(SSEEventType.STATUS, content=f"Using {friendly_name}...")
                        yield create_sse_event(SSEEventType.TOOL_CALL, tool=tool_name, args=tool_args)

                        # Execute tool
                        tool_result, error = await execute_tool_async(tool_name, tool_args, tools)
                        tool_calls_made.append({
                            "name": tool_name,
                            "args": tool_args,
                            "result": tool_result,
                        })

                    # Append tool results to messages
                    messages.append(response)
                    for tc, made in zip(response.tool_calls, tool_calls_made[-len(response.tool_calls):]):
                        messages.append(ToolMessage(
                            content=str(made["result"]),
                            tool_call_id=tc["id"],
                        ))

                    # Check for more tool calls
                    logger.info("[ArxivEngine] Checking for more tool calls...")
                    response = await model_with_tools.ainvoke(messages)
                    
                    if not response.tool_calls:
                        logger.info("[ArxivEngine] No more tool calls. Generating final response.")
                        break

                yield create_sse_event(SSEEventType.STATUS, content="Generating response...")

                # Stream final response after tools
                if response.content:
                    content = normalize_content(response.content)
                    if content:
                        for chunk_text in chunk_content(content):
                            full_response += chunk_text
                            yield create_sse_event(SSEEventType.TOKEN, content=chunk_text)
                        logger.info(f"[ArxivEngine] Yielded response. Length: {len(full_response)}")
                    else:
                        yield create_sse_event(SSEEventType.ERROR, message="Model generated empty response")
                else:
                    # Try streaming if no content
                    logger.info("[ArxivEngine] No content, attempting stream...")
                    async for chunk in self.llm.astream(messages):
                        if hasattr(chunk, "content"):
                            content = normalize_content(chunk.content)
                            if content:
                                full_response += content
                                yield create_sse_event(SSEEventType.TOKEN, content=content)
                    
                    if not full_response:
                        yield create_sse_event(SSEEventType.ERROR, message="No response after tool execution")

            else:
                # No tools needed - direct streaming
                logger.info("[ArxivEngine] No tool calls. Streaming direct response...")
                async for chunk in model_with_tools.astream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        content = normalize_content(chunk.content)
                        if content:
                            full_response += content
                            yield create_sse_event(SSEEventType.TOKEN, content=content)
                logger.info(f"[ArxivEngine] Direct stream complete. Length: {len(full_response)}")

            # Save to history
            history_query = original_query if original_query is not None else query
            self.add_to_history(session_id, history_query, full_response)
            yield create_sse_event(SSEEventType.DONE, session_id=session_id)

        except Exception as e:
            logger.error(f"[ArxivEngine] Error in query_stream: {e}", exc_info=True)
            yield create_sse_event(SSEEventType.ERROR, message=str(e))

    # =========================================================================
    # Non-streaming query
    # =========================================================================
    
    async def query(
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

        async for chunk in self.query_stream(query, session_id, arxiv_enabled, original_query):
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
