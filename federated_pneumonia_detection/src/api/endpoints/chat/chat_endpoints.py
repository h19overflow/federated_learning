from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.retriver import (
    QueryEngine,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.mcp_manager import (
    MCPManager,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent import (
    ArxivAugmentedEngine,
)
from ..schema import ChatMessage, ChatResponse, ChatHistoryResponse
from .chat_utils import enhance_query_with_run_context
import logging
import uuid
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

# Initialize QueryEngine once (may fail if PostgreSQL unavailable)
query_engine = None
try:
    query_engine = QueryEngine()
    logger.info("QueryEngine initialized successfully")
except Exception as e:
    logger.warning(f"QueryEngine initialization failed (database unavailable): {e}")

# Get MCP manager singleton
mcp_manager = MCPManager.get_instance()

# Lazy-initialized ArxivAugmentedEngine
_arxiv_engine: Optional[ArxivAugmentedEngine] = None


def get_arxiv_engine() -> ArxivAugmentedEngine:
    """Get or create ArxivAugmentedEngine instance."""
    global _arxiv_engine
    if _arxiv_engine is None:
        _arxiv_engine = ArxivAugmentedEngine()
    return _arxiv_engine


@router.post("/query", response_model=ChatResponse)
async def query_chat(message: ChatMessage) -> ChatResponse:
    """
    Query the retrieval-augmented generation system with a user message.
    Supports session-based conversation history and run-specific context.

    Args:
        message: ChatMessage containing the user query, optional session_id, and optional run context

    Returns:
        ChatResponse with answer, source documents, and session_id
    """
    if query_engine is None:
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    try:
        # Generate session_id if not provided
        session_id = message.session_id or str(uuid.uuid4())

        # Enhance query with run context if provided
        enhanced_query = message.query
        if message.run_id is not None:
            enhanced_query = enhance_query_with_run_context(
                message.query, message.run_id
            )

        # Use query_with_history for session-based queries
        result = query_engine.query_with_history(enhanced_query, session_id)

        sources = []
        if "context" in result:
            sources = [
                doc.metadata.get("source", "Unknown") for doc in result["context"]
            ]

        return ChatResponse(
            answer=result.get("answer", ""), sources=sources, session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/arxiv/status")
async def get_arxiv_status() -> Dict[str, Any]:
    """
    Check if arxiv MCP server is available.

    Returns:
        Dict with availability status and available tool names
    """
    return {
        "available": mcp_manager.is_available,
        "tools": (
            [t.name for t in mcp_manager.get_arxiv_tools()]
            if mcp_manager.is_available
            else []
        ),
    }


@router.post("/query/stream")
async def query_chat_stream(message: ChatMessage):
    """
    Stream chat response token by token using Server-Sent Events (SSE).

    Args:
        message: ChatMessage containing the user query, optional session_id,
                 optional run context, and arxiv_enabled flag

    Returns:
        StreamingResponse with SSE format containing tokens
    """
    # Use ArxivAugmentedEngine when arxiv is enabled, else use QueryEngine
    use_arxiv = message.arxiv_enabled

    if not use_arxiv and query_engine is None:
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    async def generate():
        session_id = message.session_id or str(uuid.uuid4())

        # Send session_id first so frontend knows it
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        enhanced_query = message.query
        if message.run_id is not None:
            enhanced_query = enhance_query_with_run_context(
                message.query, message.run_id
            )

        try:
            if use_arxiv:
                # Use ArxivAugmentedEngine with arxiv tools
                arxiv_engine = get_arxiv_engine()
                async for chunk in arxiv_engine.query_stream(
                    enhanced_query, session_id, arxiv_enabled=True
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Use standard QueryEngine
                async for chunk in query_engine.query_with_history_stream(
                    enhanced_query, session_id
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """
    Retrieve conversation history for a session.

    Args:
        session_id: Session identifier

    Returns:
        ChatHistoryResponse with conversation history
    """
    if query_engine is None:
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    try:
        history = query_engine.get_history(session_id)
        formatted_history = [
            {"user": user_msg, "assistant": ai_msg} for user_msg, ai_msg in history
        ]
        return ChatHistoryResponse(history=formatted_history, session_id=session_id)
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving history: {str(e)}"
        )


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str) -> Dict[str, str]:
    """
    Clear conversation history for a session.

    Args:
        session_id: Session identifier

    Returns:
        Success message
    """
    if query_engine is None:
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    try:
        query_engine.clear_history(session_id)
        return {"message": f"History cleared for session {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")


@router.post("/retrieve")
async def retrieve_documents(message: ChatMessage) -> Dict[str, Any]:
    """
    Retrieve documents based on a query without generating an answer.

    Args:
        message: ChatMessage containing the query

    Returns:
        Dict with retrieved documents and their metadata
    """
    if query_engine is None:
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    try:
        results = query_engine.query(message.query)
        documents = []
        for doc in results:
            documents.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                }
            )
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {str(e)}"
        )
