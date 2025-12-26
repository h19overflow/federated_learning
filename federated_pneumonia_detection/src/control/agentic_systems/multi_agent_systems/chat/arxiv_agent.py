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
        logger.info(f"[ArxivEngine] Initializing with max_history={max_history}")
        self.max_history = max_history
        self.conversation_history: Dict[str, List[Tuple[str, str]]] = {}

        try:
            logger.info("[ArxivEngine] Initializing ChatGoogleGenerativeAI (gemini-3-flash-preview)...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                temperature=0.7,
            )
            logger.info("[ArxivEngine] LLM initialized successfully")
        except Exception as e:
            logger.error(f"[ArxivEngine] Failed to initialize LLM: {e}", exc_info=True)
            # Reraise or handle? Original code didn't handle. Let it crash if LLM fails.
            raise e

        # Initialize RAG tool with QueryEngine (optional - may fail if DB unavailable)
        self._query_engine = None
        self._rag_tool = None
        try:
            logger.info("[ArxivEngine] Initializing RAG tool via QueryEngine...")
            self._query_engine = QueryEngine()
            self._rag_tool = create_rag_tool(self._query_engine)
            logger.info("[ArxivEngine] RAG tool initialized successfully")
        except Exception as e:
            logger.warning(f"[ArxivEngine] RAG tool unavailable (database not connected): {e}")
            logger.info("[ArxivEngine] ArxivAugmentedEngine will work with arxiv tools only")

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
        logger.info(f"[ArxivEngine] Starting stream for session {session_id}, query: '{query[:50]}...', arxiv={arxiv_enabled}")
        
        tools = self._get_tools(arxiv_enabled)
        logger.info(f"[ArxivEngine] Retrieved {len(tools)} tools: {[t.name for t in tools]}")

        if not tools:
            logger.error("[ArxivEngine] No tools available")
            yield {"type": "error", "message": "No tools available"}
            return

        try:
            # Bind tools to the model
            logger.info("[ArxivEngine] Binding tools to model")
            model_with_tools = self.llm.bind_tools(tools)

            # Build messages with history
            messages = self._build_messages(query, session_id)
            logger.info(f"[ArxivEngine] Built {len(messages)} messages including history")

            full_response = ""
            tool_calls_made = []

            # Yield initial status
            yield {"type": "status", "content": "Analyzing your query..."}

            # First call - check if model wants to use tools
            logger.info("[ArxivEngine] Invoking model to check for tool calls...")
            response = await model_with_tools.ainvoke(messages)

            # Check for tool calls - loop until model stops calling tools
            if response.tool_calls:
                while response.tool_calls:
                    logger.info(f"[ArxivEngine] Model requested {len(response.tool_calls)} tool calls")

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        logger.info(f"[ArxivEngine] Executing tool: {tool_name} with args: {tool_args}")

                        # Yield a user-friendly status message for the tool
                        friendly_name = tool_name.replace("_", " ").title()
                        yield {"type": "status", "content": f"Using {friendly_name}..."}

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
                                    # Use async invocation for tools (arxiv tools are async)
                                    tool_result = await tool.ainvoke(tool_args)
                                    logger.info(f"[ArxivEngine] Tool {tool_name} executed successfully. Result length: {len(str(tool_result))}")
                                except Exception as e:
                                    tool_result = f"Tool error: {str(e)}"
                                    logger.error(f"[ArxivEngine] Tool {tool_name} failed: {e}", exc_info=True)
                                break

                        tool_calls_made.append(
                            {
                                "name": tool_name,
                                "args": tool_args,
                                "result": tool_result,
                            }
                        )

                    # Build follow-up messages with tool results
                    logger.info("[ArxivEngine] Appending tool results to conversation")
                    from langchain_core.messages import ToolMessage

                    messages.append(response)
                    for tc, made in zip(response.tool_calls, tool_calls_made[-len(response.tool_calls):]):
                        messages.append(
                            ToolMessage(
                                content=str(made["result"]),
                                tool_call_id=tc["id"],
                            )
                        )
                    
                    # Invoke model again to check if it wants more tools or will respond
                    logger.info("[ArxivEngine] Invoking model again to check for more tool calls...")
                    response = await model_with_tools.ainvoke(messages)
                    
                    # If no more tool calls, break the loop
                    if not response.tool_calls:
                        logger.info("[ArxivEngine] No more tool calls. Model ready to generate final response.")
                        break
                
                # Yield status before generating response
                yield {"type": "status", "content": "Generating response..."}

                # Now stream the final response after all tools have been executed
                logger.info("[ArxivEngine] Streaming final response after tool execution...")
                logger.info(f"[ArxivEngine] Final response has content: {bool(response.content)}")
                
                # Check if response already has content (from the last ainvoke)
                if response.content:
                    logger.info("[ArxivEngine] Using content from last model invocation")
                    # Process the content (might be a list for Gemini)
                    content = response.content
                    if isinstance(content, list):
                        content = "".join(
                            part if isinstance(part, str) else part.get("text", "")
                            for part in content
                        )
                    
                    if content:
                        # Yield the entire response as chunks (simulate streaming)
                        # Split into smaller chunks for better UX
                        chunk_size = 50  # characters per chunk
                        for i in range(0, len(content), chunk_size):
                            chunk_text = content[i:i+chunk_size]
                            full_response += chunk_text
                            yield {"type": "token", "content": chunk_text}
                        
                        logger.info(f"[ArxivEngine] Yielded response in {len(content)//chunk_size + 1} chunks. Response length: {len(full_response)}")
                    else:
                        logger.error("[ArxivEngine] Response content is empty after processing")
                        yield {"type": "error", "message": "Model generated empty response"}
                else:
                    # Try streaming if no content in response
                    logger.info("[ArxivEngine] No content in response, attempting to stream...")
                    chunk_count = 0
                    async for chunk in self.llm.astream(messages):
                        if hasattr(chunk, "content"):
                            content = chunk.content
                            
                            # Process list content (Gemini format)
                            if isinstance(content, list):
                                content = "".join(
                                    part if isinstance(part, str) else part.get("text", "")
                                    for part in content
                                )
                            
                            if content:
                                full_response += content
                                chunk_count += 1
                                yield {"type": "token", "content": content}
                    
                    if chunk_count == 0:
                        logger.error("[ArxivEngine] Streaming produced no content")
                        yield {"type": "error", "message": "Model generated no response after tool execution"}
                
                logger.info(f"[ArxivEngine] Stream completed (with tools). Response length: {len(full_response)}")

            else:
                # No tools needed, stream direct response
                logger.info("[ArxivEngine] No tool calls requested. Streaming direct response...")
                chunk_count = 0
                async for chunk in model_with_tools.astream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        # Handle content that may be a list (Gemini) or string
                        content = chunk.content
                        if isinstance(content, list):
                            content = "".join(
                                part if isinstance(part, str) else part.get("text", "")
                                for part in content
                            )
                        
                        if content:
                            full_response += content
                            chunk_count += 1
                            yield {"type": "token", "content": content}
                logger.info(f"[ArxivEngine] Stream completed (direct). Generated {chunk_count} chunks. Response length: {len(full_response)}")


            # Save to history
            self.add_to_history(session_id, query, full_response)
            yield {"type": "done", "session_id": session_id}

        except Exception as e:
            logger.error(f"[ArxivEngine] Error in query_stream: {e}", exc_info=True)
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
