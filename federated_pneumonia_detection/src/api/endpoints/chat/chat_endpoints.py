from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.retriver import (
    QueryEngine,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.mcp_manager import (
    MCPManager,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent import (
    ArxivAugmentedEngine,
)
from ..schema import ChatMessage, ChatResponse, ChatHistoryResponse, ChatSessionSchema
from .chat_utils import enhance_query_with_run_context
from federated_pneumonia_detection.src.boundary.CRUD.chat_history import (
    get_all_chat_sessions,
    create_chat_session,
    delete_chat_session,
    get_chat_session
)

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


@router.get("/sessions", response_model=List[ChatSessionSchema])
async def list_chat_sessions():
    """List all available chat sessions."""
    sessions = get_all_chat_sessions()
    # Format dates as strings
    return [
        ChatSessionSchema(
            id=s.id,
            title=s.title,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat()
        ) for s in sessions
    ]


@router.post("/sessions", response_model=ChatSessionSchema)
async def create_new_chat_session(title: Optional[str] = None):
    """Create a new chat session."""
    session = create_chat_session(title=title)
    return ChatSessionSchema(
        id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat()
    )


@router.delete("/sessions/{session_id}")
async def delete_existing_chat_session(session_id: str):
    """Delete a chat session."""
    success = delete_chat_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Also clear history in the agent for this session if it's in memory/persistent store
    arxiv_engine = get_arxiv_engine()
    arxiv_engine.clear_history(session_id)
    
    return {"message": f"Session {session_id} deleted"}


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
        
        # Ensure session exists in DB
        existing_session = get_chat_session(session_id)
        if not existing_session:
            create_chat_session(title=message.query[:50] + "...", session_id=session_id)

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
    logger.info(f"[STREAM] Received streaming query request - Query: '{message.query[:100]}...', "
                f"Session ID: {message.session_id}, Run ID: {message.run_id}, "
                f"Arxiv Enabled: {message.arxiv_enabled}")
    
    # Use ArxivAugmentedEngine when arxiv is enabled, else use QueryEngine
    use_arxiv = message.arxiv_enabled
    logger.info(f"[STREAM] Using {'ArxivAugmentedEngine' if use_arxiv else 'QueryEngine'}")

    if not use_arxiv and query_engine is None:
        logger.error("[STREAM] QueryEngine not initialized - cannot process request")
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    async def generate():
        try:
            logger.info("[STREAM] Starting stream generation")
            
            # Generate session_id
            session_id = message.session_id or str(uuid.uuid4())
            logger.info(f"[STREAM] Session ID: {session_id}")

            # Ensure session exists in DB
            try:
                existing_session = get_chat_session(session_id)
                if not existing_session:
                    logger.info(f"[STREAM] Creating new session in DB: {session_id}")
                    # Passing session_id to create_chat_session as well (need to update CRUD)
                    create_chat_session(title=message.query[:50] + "...", session_id=session_id)
            except Exception as e:
                logger.warning(f"[STREAM] Failed to ensure session exists in DB: {e}")

            # Send session_id first so frontend knows it
            session_data = {'type': 'session', 'session_id': session_id}
            logger.debug(f"[STREAM] Sending session data: {session_data}")
            yield f"data: {json.dumps(session_data)}\n\n"

            # Enhance query with run context if provided
            enhanced_query = message.query
            if message.run_id is not None:
                logger.info(f"[STREAM] Enhancing query with run context for run_id: {message.run_id}")
                try:
                    enhanced_query = enhance_query_with_run_context(
                        message.query, message.run_id
                    )
                    logger.info(f"[STREAM] Query enhanced successfully")
                except Exception as e:
                    logger.error(f"[STREAM] Failed to enhance query with run context: {e}", exc_info=True)
                    # Continue with original query if enhancement fails
                    enhanced_query = message.query
            
            logger.info(f"[STREAM] Final query to process: '{enhanced_query[:100]}...'")

            chunk_count = 0
            try:
                if use_arxiv:
                    # Use ArxivAugmentedEngine with arxiv tools
                    logger.info("[STREAM] Initializing ArxivAugmentedEngine")
                    try:
                        arxiv_engine = get_arxiv_engine()
                        logger.info("[STREAM] ArxivAugmentedEngine initialized successfully")
                    except Exception as e:
                        logger.error(f"[STREAM] Failed to initialize ArxivAugmentedEngine: {e}", exc_info=True)
                        error_data = {'type': 'error', 'message': f'Failed to initialize Arxiv engine: {str(e)}'}
                        yield f"data: {json.dumps(error_data)}\n\n"
                        return
                    
                    logger.info("[STREAM] Starting arxiv query stream")
                    async for chunk in arxiv_engine.query_stream(
                        enhanced_query, session_id, arxiv_enabled=True
                    ):
                        chunk_count += 1
                        if chunk_count % 10 == 0:  # Log every 10th chunk to avoid spam
                            logger.debug(f"[STREAM] Arxiv chunk #{chunk_count}: {chunk.get('type', 'unknown')}")
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    logger.info(f"[STREAM] Arxiv stream completed - Total chunks: {chunk_count}")
                else:
                    # Use standard QueryEngine
                    logger.info("[STREAM] Starting standard query stream")
                    try:
                        async for chunk in query_engine.query_with_history_stream(
                            enhanced_query, session_id
                        ):
                            chunk_count += 1
                            if chunk_count % 10 == 0:  # Log every 10th chunk to avoid spam
                                logger.debug(f"[STREAM] Standard chunk #{chunk_count}: {chunk.get('type', 'unknown')}")
                            yield f"data: {json.dumps(chunk)}\n\n"
                        
                        logger.info(f"[STREAM] Standard stream completed - Total chunks: {chunk_count}")
                    except Exception as e:
                        logger.error(f"[STREAM] Error in query_with_history_stream: {e}", exc_info=True)
                        error_data = {'type': 'error', 'message': f'Query engine error: {str(e)}'}
                        yield f"data: {json.dumps(error_data)}\n\n"
                        return
                
                if chunk_count == 0:
                    logger.warning("[STREAM] No chunks were generated - this may cause empty message on frontend")
                    error_data = {'type': 'error', 'message': 'No response generated from the query engine'}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    
            except Exception as e:
                logger.error(f"[STREAM] Error during streaming iteration: {e}", exc_info=True)
                error_data = {'type': 'error', 'message': f'Streaming error: {str(e)}'}
                yield f"data: {json.dumps(error_data)}\n\n"
                
        except Exception as e:
            logger.error(f"[STREAM] Critical error in generate function: {e}", exc_info=True)
            error_data = {'type': 'error', 'message': f'Critical streaming error: {str(e)}'}
            yield f"data: {json.dumps(error_data)}\n\n"

    logger.info("[STREAM] Returning StreamingResponse")
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
        arxiv_engine = get_arxiv_engine()
        history = arxiv_engine.get_history(session_id)
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
        arxiv_engine = get_arxiv_engine()
        arxiv_engine.clear_history(session_id)
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
