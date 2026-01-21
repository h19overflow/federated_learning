"""Helper utilities for research agent composition."""

from __future__ import annotations

import logging
from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.arxiv_mcp import (
    MCPManager,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.tools.arxiv_embedding_tool import (
    create_arxiv_embedding_tool,
)

logger = logging.getLogger(__name__)


def build_tools(rag_tool, arxiv_enabled: bool) -> list:
    """Build tool list based on arxiv availability."""
    tools = []
    if rag_tool is not None:
        tools.append(rag_tool)

    if not arxiv_enabled:
        return tools

    mcp_manager = MCPManager.get_instance()
    if not mcp_manager.is_available:
        logger.warning("Arxiv requested but MCP manager not available")
        return tools

    arxiv_tools = mcp_manager.get_arxiv_tools()
    tools.extend(arxiv_tools)
    embedding_tool = create_arxiv_embedding_tool()
    tools.append(embedding_tool)
    logger.info("Added %s arxiv tools", len(arxiv_tools))
    return tools


def create_research_agent(llm, tools: list, system_prompt: str):
    """Create a LangChain agent with tools for research orchestration."""
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def build_messages(
    history_manager,
    query: str,
    session_id: str,
    include_system: bool,
    system_prompt: str,
) -> List[SystemMessage | HumanMessage | AIMessage]:
    """Build the message list for the agent."""
    messages: List[SystemMessage | HumanMessage | AIMessage] = []
    if include_system:
        messages.append(SystemMessage(content=system_prompt))
    messages.extend(history_manager.get_messages(session_id))
    messages.append(HumanMessage(content=query))
    return messages
