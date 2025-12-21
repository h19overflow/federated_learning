"""
RAG Tool - LangChain tool wrapper for local knowledge base search.

Wraps QueryEngine.query() to allow agent access to local research papers.

Dependencies:
    - QueryEngine from retriver module
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.retriver import (
        QueryEngine,
    )

logger = logging.getLogger(__name__)


class RAGToolInput(BaseModel):
    """Input schema for RAG tool."""

    query: str = Field(description="Search query for local documents")


_query_engine: Optional["QueryEngine"] = None


def set_query_engine(engine: "QueryEngine") -> None:
    """
    Set the QueryEngine instance for the RAG tool.

    Args:
        engine: Initialized QueryEngine instance
    """
    global _query_engine
    _query_engine = engine
    logger.info("RAG tool query engine configured")


def _format_results(documents: list) -> str:
    """
    Format retrieved documents as a readable string.

    Args:
        documents: List of LangChain Document objects

    Returns:
        Formatted string with document excerpts and metadata
    """
    if not documents:
        return "No relevant documents found in the local knowledge base."

    formatted_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content[:500]  # Truncate for readability

        formatted_parts.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {content}...\n"
        )

    return "\n---\n".join(formatted_parts)


@tool(args_schema=RAGToolInput)
def search_local_knowledge_base(query: str) -> str:
    """
    Search the local knowledge base of uploaded research papers.

    Searches through federated learning and medical imaging research papers
    stored in the local vector database using hybrid BM25 + semantic search.

    Args:
        query: Search query for local documents

    Returns:
        Formatted string of retrieved document excerpts with metadata
    """
    if _query_engine is None:
        logger.error("RAG tool called but QueryEngine not initialized")
        return "Error: Local knowledge base is not available."

    try:
        results = _query_engine.query(query)
        return _format_results(results)

    except Exception as e:
        logger.error(f"RAG tool query failed: {e}")
        return f"Error searching local knowledge base: {str(e)}"


def create_rag_tool(query_engine: "QueryEngine") -> BaseTool:
    """
    Create a RAG tool with the provided QueryEngine.

    Factory function that configures the global query engine and returns
    the tool for agent use.

    Args:
        query_engine: Initialized QueryEngine instance

    Returns:
        Configured search_local_knowledge_base tool
    """
    set_query_engine(query_engine)
    return search_local_knowledge_base
