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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent_prompts import (
    RESEARCH_MODE_SYSTEM_PROMPT,
    BASIC_MODE_SYSTEM_PROMPT,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent.query_router import (
    classify_query,
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

from .content import normalize_content
from .history import ChatHistoryManager
from .streaming import SSEEventType, create_sse_event
from dotenv import load_dotenv

load_dotenv()
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
                model="gemini-3-pro-preview",
                temperature=1.0,  # Gemini 3 default - prevents infinite loops
                max_tokens=2048,  # Limit response length for concise answers
                thinking_level="low",  # Fast reasoning for research queries
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
            else:
                logger.warning("Arxiv requested but MCP manager not available")

        return tools

    def _create_agent(self, tools: list, system_prompt: str):
        """
        Create a LangChain agent with tools for research orchestration.

        Args:
            tools: List of LangChain tools for the agent
            system_prompt: System prompt defining agent behavior

        Returns:
            Configured agent for tool-augmented generation
        """
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
        )
        return agent

    # =========================================================================
    # Message building
    # =========================================================================
    
    def _build_messages(
        self, query: str, session_id: str, mode: str = "research"
    ) -> List[SystemMessage | HumanMessage | AIMessage]:
        """
        Build message list for the agent.

        Args:
            query: Current user query
            session_id: Session identifier for history
            mode: Query mode - "research" for tool-augmented, "basic" for conversational

        Returns:
            List of messages for the agent
        """
        # Select system prompt based on mode
        system_prompt = (
            RESEARCH_MODE_SYSTEM_PROMPT if mode == "research" else BASIC_MODE_SYSTEM_PROMPT
        )
        messages = [SystemMessage(content=system_prompt)]

        # Add history from persistent store
        messages.extend(self._history_manager.get_messages(session_id))

        # Add current query
        messages.append(HumanMessage(content=query))

        return messages
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
            query: User's question (may include run context enhancement)
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


        classification_query = original_query if original_query else query
        query_mode = classify_query(classification_query)
        logger.debug(f"[ArxivEngine] Query classified as: {query_mode} (using {'original' if original_query else 'full'} query)")

        # Step 2: Conditionally load tools based on mode
        if query_mode == "research":
            tools = self._get_tools(arxiv_enabled)
            logger.debug(f"[ArxivEngine] Retrieved {len(tools)} tools: {[t.name for t in tools]}")

            if not tools:
                logger.error("[ArxivEngine] No tools available")
                yield create_sse_event(SSEEventType.ERROR, message="No tools available")
                return

        try:
            # Build messages with appropriate prompt
            messages = self._build_messages(query, session_id, mode=query_mode)
            logger.debug(f"[ArxivEngine] Built {len(messages)} messages including history")

            full_response = ""
            tool_calls_made = []

            # Step 3: Handle based on query mode
            if query_mode == "basic":
                # Basic mode: Direct streaming without tools
                logger.debug("[ArxivEngine] Basic mode - streaming direct response...")
                chunk_count = 0
                try:
                    async for chunk in self.llm.astream(messages):
                        chunk_count += 1
                        if hasattr(chunk, "content") and chunk.content:
                            content = normalize_content(chunk.content)
                            if content:
                                full_response += content
                                yield create_sse_event(SSEEventType.TOKEN, content=content)
                    logger.debug(f"[ArxivEngine] Basic mode complete. Chunks: {chunk_count}, Length: {len(full_response)}")

                    if not full_response:
                        logger.error(f"[ArxivEngine] Basic mode produced no response. Chunks received: {chunk_count}")
                        yield create_sse_event(SSEEventType.ERROR, message="No response generated. Please try again.")
                        return
                except Exception as e:
                    logger.error(f"[ArxivEngine] Exception in basic mode streaming: {e}", exc_info=True)
                    yield create_sse_event(SSEEventType.ERROR, message=f"Streaming failed: {str(e)}")
                    return

            else:
                # Research mode: Tool-augmented generation with agent
                yield create_sse_event(SSEEventType.STATUS, content="Analyzing your query...")

                # Create agent with tools
                logger.info("[ArxivEngine] Creating agent with tools for research mode...")
                agent = self._create_agent(tools, RESEARCH_MODE_SYSTEM_PROMPT)

                # Build input dict for agent
                agent_input = {"messages": messages}
                logger.info(f"[ArxivEngine] Invoking agent with {len(messages)} messages...")

                # Stream agent response
                try:
                    chunk_count = 0
                    async for event in agent.astream(agent_input):
                        chunk_count += 1

                        # Handle agent event types
                        if "messages" in event:
                            # Final or intermediate messages from agent
                            for msg in event["messages"]:
                                # Tool call events
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        tool_name = tool_call.get("name", "unknown")
                                        tool_args = tool_call.get("args", {})
                                        logger.info(f"[ArxivEngine] Agent using tool: {tool_name}")
                                        # Yield user-friendly status
                                        friendly_name = tool_name.replace("_", " ").title()
                                        yield create_sse_event(SSEEventType.STATUS, content=f"Using {friendly_name}...")
                                        yield create_sse_event(SSEEventType.TOOL_CALL, tool=tool_name, args=tool_args)

                                        tool_calls_made.append({
                                            "name": tool_name,
                                            "args": tool_args,
                                        })

                                # Content streaming from agent
                                if hasattr(msg, "content") and msg.content:
                                    if isinstance(msg, AIMessage):
                                        content = normalize_content(msg.content)
                                        if content:
                                            full_response += content
                                            yield create_sse_event(SSEEventType.TOKEN, content=content)

                    logger.info(f"[ArxivEngine] Agent stream complete. Events: {chunk_count}, Response length: {len(full_response)}")

                    # If no response collected from events, try extracting from final state
                    if not full_response:
                        logger.warning("[ArxivEngine] No response in agent stream. This may indicate agent didn't generate text.")
                        yield create_sse_event(SSEEventType.ERROR, message="Agent completed but produced no response. Please try rephrasing.")
                        return

                except Exception as e:
                    logger.error(f"[ArxivEngine] Exception during agent streaming: {e}", exc_info=True)
                    yield create_sse_event(SSEEventType.ERROR, message=f"Agent execution failed: {str(e)}")
                    return

            # Save to history
            try:
                history_query = original_query if original_query is not None else query
                self.add_to_history(session_id, history_query, full_response)
                logger.debug(f"[ArxivEngine] Saved to history. Session: {session_id}, Query length: {len(history_query)}, Response length: {len(full_response)}")
            except Exception as e:
                logger.error(f"[ArxivEngine] Failed to save to history: {e}", exc_info=True)
                # Non-fatal - we already have the response

            yield create_sse_event(SSEEventType.DONE, session_id=session_id)
            logger.info(f"[ArxivEngine] Stream complete for session {session_id}")

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
