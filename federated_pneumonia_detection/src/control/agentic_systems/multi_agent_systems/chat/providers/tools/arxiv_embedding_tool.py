"""
Arxiv Embedding Tool - LangChain tool for embedding arxiv papers into knowledge base.

Downloads arxiv papers via MCP and inserts into PGVector for RAG retrieval.

Dependencies:
    - MCPManager for arxiv download
    - PGVector for storage
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from pydantic import BaseModel, Field

from federated_pneumonia_detection.src.boundary.engine import settings

logger = logging.getLogger(__name__)


class EmbedArxivPaperInput(BaseModel):
    """Input schema for arxiv embedding tool."""

    paper_id: str = Field(
        description="The arxiv paper ID to embed (e.g., '1706.03762' for 'Attention Is All You Need')"
    )


# Module-level singleton for PGVector connection
_vectorstore: Optional[PGVector] = None


def _get_vectorstore() -> PGVector:
    """Get or create the PGVector instance."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = PGVector(
            connection=settings.get_postgres_db_uri(),
            collection_name="research_papers",
            embeddings=embeddings,
        )
    return _vectorstore


async def _download_paper_via_mcp(paper_id: str) -> Dict[str, Any]:
    """
    Download paper using MCP arxiv server.

    Args:
        paper_id: Arxiv paper ID

    Returns:
        Dict with 'status', 'path', and optionally 'error'
    """
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.providers.arxiv_mcp import (
        MCPManager,
    )

    manager = MCPManager.get_instance()

    # Ensure MCP Manager is initialized before checking availability
    if not manager.is_available:
        logger.info(
            "[EmbedTool] MCP Manager not initialized, attempting initialization..."
        )
        try:
            await manager.initialize()
        except Exception as e:
            logger.error(
                f"[EmbedTool] Failed to initialize MCP Manager: {e}", exc_info=True
            )
            return {
                "status": "error",
                "error": f"Failed to initialize MCP Manager: {str(e)}",
            }

    if not manager.is_available:
        return {
            "status": "error",
            "error": "MCP Manager not available after initialization attempt",
        }

    tools = manager.get_arxiv_tools()
    download_tool = next((t for t in tools if t.name == "download_paper"), None)

    if not download_tool:
        return {"status": "error", "error": "download_paper tool not found"}

    try:
        result = await download_tool.ainvoke({"paper_id": paper_id})
        logger.info(f"[EmbedTool] Raw download result type: {type(result)}")
        logger.info(f"[EmbedTool] Raw download result: {str(result)[:500]}...")

        # Parse result - it's a list with a dict containing 'text' key
        text_content = ""
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and "text" in item:
                    text_content = item["text"]
                    break
        elif isinstance(result, str):
            text_content = result

        logger.info(f"[EmbedTool] Extracted text content: {text_content[:200]}...")

        if text_content:
            data = json.loads(text_content)
            logger.info(f"[EmbedTool] Parsed JSON data keys: {data.keys()}")
            return data
        else:
            return {"status": "error", "error": "Empty response from download tool"}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse download result: {e}")
        return {"status": "error", "error": f"JSON parse error: {str(e)}"}
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


def _read_markdown_file(path: str) -> str:
    """Read content from markdown file path."""
    # Handle file:// URLs
    if path.startswith("file://"):
        path = path[7:]  # Remove file:// prefix

    # Normalize path for Windows
    path = path.replace("\\\\", "\\").replace("/", os.sep)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Paper not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _chunk_content(content: str, paper_id: str) -> list[Document]:
    """
    Split markdown content into chunks for embedding.

    Args:
        content: Markdown text content
        paper_id: Arxiv paper ID for metadata

    Returns:
        List of Document objects with metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    chunks = splitter.split_text(content)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": f"arxiv:{paper_id}",
                "paper_id": paper_id,
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    return documents


async def embed_arxiv_paper_async(paper_id: str) -> Dict[str, Any]:
    """
    Core async logic for embedding an arxiv paper.

    Args:
        paper_id: Arxiv paper ID

    Returns:
        Dict with status and details
    """
    logger.info(f"[EmbedTool] Starting embedding for paper {paper_id}")

    # Step 1: Download paper via MCP
    download_result = await _download_paper_via_mcp(paper_id)
    logger.info(f"[EmbedTool] Download result: {download_result}")

    if download_result.get("status") == "error":
        error_msg = download_result.get("error", "Download failed")
        logger.error(f"[EmbedTool] Download failed: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
        }

    # Step 2: Get file path (MCP returns 'resource_uri' instead of 'path')
    file_path = download_result.get("resource_uri") or download_result.get("path")
    logger.info(f"[EmbedTool] File path from download: {file_path}")
    if not file_path:
        logger.error(
            f"[EmbedTool] No 'resource_uri' or 'path' key in download_result. Keys present: {download_result.keys()}"
        )
        return {
            "success": False,
            "error": "No file path returned from download",
        }

    logger.info(f"[EmbedTool] Paper downloaded to: {file_path}")

    # Step 3: Read content
    try:
        content = _read_markdown_file(file_path)
        logger.info(f"[EmbedTool] Read {len(content)} characters from file")
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Failed to read file: {str(e)}"}

    # Step 4: Chunk content
    documents = _chunk_content(content, paper_id)
    logger.info(f"[EmbedTool] Created {len(documents)} chunks")

    # Step 5: Insert into vector store
    try:
        vectorstore = _get_vectorstore()
        logger.info(
            f"[EmbedTool] Inserting {len(documents)} chunks into vectorstore..."
        )

        # Add documents to the vector store
        ids = vectorstore.add_documents(documents)

        logger.info(
            f"[EmbedTool] Successfully embedded {len(documents)} chunks for {paper_id}"
        )
        logger.info(
            f"[EmbedTool] Document IDs: {ids[:3]}..."
            if len(ids) > 3
            else f"[EmbedTool] Document IDs: {ids}"
        )

    except Exception as e:
        logger.error(
            f"[EmbedTool] Failed to insert into vectorstore: {e}", exc_info=True
        )
        return {"success": False, "error": f"Vector store insertion failed: {str(e)}"}

    return {
        "success": True,
        "paper_id": paper_id,
        "chunks_embedded": len(documents),
        "message": f"Successfully embedded paper {paper_id} with {len(documents)} chunks into the knowledge base.",
    }


@tool(args_schema=EmbedArxivPaperInput)
async def embed_arxiv_paper(paper_id: str) -> str:
    """
    Download and embed an arxiv paper into the local knowledge base.

    This tool downloads a research paper from arxiv, converts it to text,
    splits it into chunks, and stores it in the vector database for future
    retrieval. Use this when the user explicitly requests to add a paper
    to the knowledge base.

    IMPORTANT: Always ask the user for confirmation before using this tool,
    as it permanently adds content to the knowledge base.

    Args:
        paper_id: The arxiv paper ID (e.g., '1706.03762')

    Returns:
        Success or error message
    """
    result = await embed_arxiv_paper_async(paper_id)

    if result.get("success"):
        return result["message"]
    else:
        return (
            f"Failed to embed paper {paper_id}: {result.get('error', 'Unknown error')}"
        )


def create_arxiv_embedding_tool():
    """
    Create the arxiv embedding tool.

    Returns:
        The embed_arxiv_paper tool ready for agent use
    """
    return embed_arxiv_paper
