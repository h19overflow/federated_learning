from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.retriver import (
    QueryEngine,
)
from .chat_utils import enhance_query_with_run_context
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

# Initialize QueryEngine once
try:
    query_engine = QueryEngine()
except Exception as e:
    logger.error(f"Error initializing QueryEngine: {e}")
    query_engine = None


class ChatMessage(BaseModel):
    query: str
    session_id: Optional[str] = None
    run_id: Optional[int] = None
    training_mode: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    session_id: str


class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, str]]
    session_id: str


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
