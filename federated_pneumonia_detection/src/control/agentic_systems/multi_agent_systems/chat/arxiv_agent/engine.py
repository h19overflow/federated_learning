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
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.tools.arxiv_embedding_tool import (
    create_arxiv_embedding_tool,
)

from .content import chunk_content, normalize_content
from .history import ChatHistoryManager
from .streaming import SSEEventType, create_sse_event, execute_tool_async
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

        # Step 1: Classify query to determine if tools are needed
        query_mode = classify_query(query)
        logger.info(f"[ArxivEngine] Query classified as: {query_mode}")

        # Step 2: Conditionally load tools based on mode
        if query_mode == "research":
            tools = self._get_tools(arxiv_enabled)
            logger.info(f"[ArxivEngine] Retrieved {len(tools)} tools: {[t.name for t in tools]}")

            if not tools:
                logger.error("[ArxivEngine] No tools available")
                yield create_sse_event(SSEEventType.ERROR, message="No tools available")
                return

        try:
            # Build messages with appropriate prompt
            messages = self._build_messages(query, session_id, mode=query_mode)
            logger.info(f"[ArxivEngine] Built {len(messages)} messages including history")

            full_response = ""
            tool_calls_made = []

            # Step 3: Handle based on query mode
            if query_mode == "basic":
                # Basic mode: Direct streaming without tools
                logger.info("[ArxivEngine] Basic mode - streaming direct response...")
                chunk_count = 0
                try:
                    async for chunk in self.llm.astream(messages):
                        chunk_count += 1
                        if hasattr(chunk, "content") and chunk.content:
                            content = normalize_content(chunk.content)
                            if content:
                                full_response += content
                                yield create_sse_event(SSEEventType.TOKEN, content=content)
                    logger.info(f"[ArxivEngine] Basic mode complete. Chunks: {chunk_count}, Length: {len(full_response)}")

                    if not full_response:
                        logger.error(f"[ArxivEngine] Basic mode produced no response. Chunks received: {chunk_count}")
                        yield create_sse_event(SSEEventType.ERROR, message="No response generated. Please try again.")
                        return
                except Exception as e:
                    logger.error(f"[ArxivEngine] Exception in basic mode streaming: {e}", exc_info=True)
                    yield create_sse_event(SSEEventType.ERROR, message=f"Streaming failed: {str(e)}")
                    return

            else:
                # Research mode: Tool-augmented generation
                yield create_sse_event(SSEEventType.STATUS, content="Analyzing your query...")

                # Bind tools to model
                model_with_tools = self.llm.bind_tools(tools)

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
                            logger.info(f"[ArxivEngine] Executing tool: {tool_name} with args: {tool_args}")

                            # Yield user-friendly status
                            friendly_name = tool_name.replace("_", " ").title()
                            yield create_sse_event(SSEEventType.STATUS, content=f"Using {friendly_name}...")
                            yield create_sse_event(SSEEventType.TOOL_CALL, tool=tool_name, args=tool_args)

                            # Execute tool
                            try:
                                tool_result, error = await execute_tool_async(tool_name, tool_args, tools)
                                if error:
                                    logger.error(f"[ArxivEngine] Tool {tool_name} returned error: {error}")
                                    yield create_sse_event(SSEEventType.ERROR, message=f"Tool {tool_name} failed: {error}")
                                    return
                                logger.info(f"[ArxivEngine] Tool {tool_name} succeeded. Result length: {len(str(tool_result))}")
                                tool_calls_made.append({
                                    "name": tool_name,
                                    "args": tool_args,
                                    "result": tool_result,
                                })
                            except Exception as e:
                                logger.error(f"[ArxivEngine] Exception executing tool {tool_name}: {e}", exc_info=True)
                                yield create_sse_event(SSEEventType.ERROR, message=f"Tool execution failed: {str(e)}")
                                return

                        # Append tool results to messages
                        try:
                            messages.append(response)
                            logger.info(f"[ArxivEngine] Appending {len(response.tool_calls)} tool results to messages")
                            for tc, made in zip(response.tool_calls, tool_calls_made[-len(response.tool_calls):]):
                                tool_msg = ToolMessage(
                                    content=str(made["result"]),
                                    tool_call_id=tc["id"],
                                )
                                messages.append(tool_msg)
                                logger.debug(f"[ArxivEngine] Added ToolMessage for {made['name']}, content length: {len(str(made['result']))}")
                        except Exception as e:
                            logger.error(f"[ArxivEngine] Failed to append tool results to messages: {e}", exc_info=True)
                            yield create_sse_event(SSEEventType.ERROR, message=f"Internal error processing tool results: {str(e)}")
                            return

                        # Check for more tool calls
                        logger.info("[ArxivEngine] Checking for more tool calls...")
                        try:
                            response = await model_with_tools.ainvoke(messages)
                            logger.info(f"[ArxivEngine] Model invoked. Has tool_calls: {bool(response.tool_calls)}, Has content: {bool(response.content)}")
                        except Exception as e:
                            logger.error(f"[ArxivEngine] Failed to invoke model for additional tool calls: {e}", exc_info=True)
                            yield create_sse_event(SSEEventType.ERROR, message=f"Model invocation failed: {str(e)}")
                            return

                        if not response.tool_calls:
                            logger.info("[ArxivEngine] No more tool calls. Generating final response.")
                            break

                    yield create_sse_event(SSEEventType.STATUS, content="Generating response...")

                    # Stream final response after tools
                    try:
                        if response.content:
                            logger.info(f"[ArxivEngine] Response has content. Length: {len(response.content)}")
                            content = normalize_content(response.content)
                            if content:
                                logger.info(f"[ArxivEngine] Normalized content length: {len(content)}")
                                chunk_count = 0
                                for chunk_text in chunk_content(content):
                                    full_response += chunk_text
                                    yield create_sse_event(SSEEventType.TOKEN, content=chunk_text)
                                    chunk_count += 1
                                logger.info(f"[ArxivEngine] Yielded {chunk_count} chunks. Total response length: {len(full_response)}")
                            else:
                                logger.error("[ArxivEngine] Content normalized to empty string")
                                yield create_sse_event(SSEEventType.ERROR, message="Model generated empty response after normalization")
                                return
                        else:
                            # Try streaming if no content
                            logger.info("[ArxivEngine] No content in response, attempting astream...")
                            logger.debug(f"[ArxivEngine] Messages count before astream: {len(messages)}")

                            chunk_count = 0
                            try:
                                async for chunk in self.llm.astream(messages):
                                    chunk_count += 1
                                    if chunk_count <= 3:
                                        logger.debug(f"[ArxivEngine] Chunk {chunk_count}: {chunk}")

                                    if hasattr(chunk, "content"):
                                        content = normalize_content(chunk.content)
                                        if content:
                                            full_response += content
                                            yield create_sse_event(SSEEventType.TOKEN, content=content)

                                logger.info(f"[ArxivEngine] Astream complete. Received {chunk_count} chunks, full_response length: {len(full_response)}")
                            except Exception as e:
                                logger.error(f"[ArxivEngine] Exception during astream: {e}", exc_info=True)
                                yield create_sse_event(SSEEventType.ERROR, message=f"Streaming error: {str(e)}")
                                return

                            if not full_response:
                                logger.error(f"[ArxivEngine] No response after tool execution. Chunks received: {chunk_count}, Messages: {len(messages)}")
                                yield create_sse_event(SSEEventType.ERROR, message="No response generated after tool execution. Please try rephrasing your question.")
                                return
                    except Exception as e:
                        logger.error(f"[ArxivEngine] Exception during response generation: {e}", exc_info=True)
                        yield create_sse_event(SSEEventType.ERROR, message=f"Response generation failed: {str(e)}")
                        return

                else:
                    # No tools needed - direct streaming
                    logger.info("[ArxivEngine] No tool calls. Streaming direct response...")
                    chunk_count = 0
                    try:
                        async for chunk in model_with_tools.astream(messages):
                            chunk_count += 1
                            if hasattr(chunk, "content") and chunk.content:
                                content = normalize_content(chunk.content)
                                if content:
                                    full_response += content
                                    yield create_sse_event(SSEEventType.TOKEN, content=content)
                        logger.info(f"[ArxivEngine] Direct stream complete. Chunks: {chunk_count}, Length: {len(full_response)}")

                        if not full_response:
                            logger.error(f"[ArxivEngine] Direct stream produced no response. Chunks received: {chunk_count}")
                            yield create_sse_event(SSEEventType.ERROR, message="No response generated. Please try again.")
                            return
                    except Exception as e:
                        logger.error(f"[ArxivEngine] Exception in direct streaming: {e}", exc_info=True)
                        yield create_sse_event(SSEEventType.ERROR, message=f"Streaming failed: {str(e)}")
                        return

            # Save to history
            try:
                history_query = original_query if original_query is not None else query
                self.add_to_history(session_id, history_query, full_response)
                logger.info(f"[ArxivEngine] Saved to history. Session: {session_id}, Query length: {len(history_query)}, Response length: {len(full_response)}")
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
