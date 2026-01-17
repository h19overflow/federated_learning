"""
Chat Endpoints Module
=====================

Provides REST API endpoints for chat functionality including streaming responses,
session management, and conversation history.

Architecture Notes
------------------
1. **Inner Generator Pattern**: Streaming endpoints use an inner `generate()` function
   because `yield` can only exist inside a generator. The outer endpoint returns a
   `StreamingResponse(generate())` which lazily consumes the generator as the client reads.

2. **Helper Functions**: Common logic (SSE formatting, DB session checks, query enhancement)
   is extracted into module-level helpers. This moves try/except blocks out of the main flow,
   keeping the streaming logic flat and readable.

3. **Guard Clauses**: Instead of deeply nested if/else blocks, we use early `return` statements
   after yielding errors. This keeps the "happy path" at a low indentation level.

4. **Closure**: The inner `generate()` function has access to the outer function's variables
   (like `message`, `use_arxiv`) without needing them passed as arguments.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
from ..schema import (
    ChatMessage,
    ChatHistoryResponse,
    ChatSessionSchema,
    CreateSessionRequest,
)
from .chat_utils import (
    sse_pack,
    sse_error,
    ensure_db_session,
    prepare_enhanced_query,
)
from ...deps import get_query_engine, get_mcp_manager, get_arxiv_engine
from federated_pneumonia_detection.src.boundary.CRUD.chat_history import (
    get_all_chat_sessions,
    create_chat_session,
    delete_chat_session
)

import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


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
async def create_new_chat_session(request: CreateSessionRequest = CreateSessionRequest()):
    """
    Create a new chat session with optional auto-generated title.

    Args:
        request: Session creation request with optional title or initial_query
    """
    title = request.title

    # Generate title from initial query if provided and no explicit title
    if request.initial_query and not title:
        from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.title_generator import (
            generate_chat_title,
        )
        title = generate_chat_title(request.initial_query)
        logger.info(f"[Session] Generated title from query: '{title}'")

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
    get_arxiv_engine().clear_history(session_id)

    return {"message": f"Session {session_id} deleted"}



@router.get("/arxiv/status")
async def get_arxiv_status() -> Dict[str, Any]:
    """
    Check if arxiv MCP server is available.

    Returns:
        Dict with availability status and available tool names
    """
    mcp_manager = get_mcp_manager()
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
    logger.info(f"[STREAM] Query: '{message.query[:100]}...' "
                f"(Session: {message.session_id}, Run: {message.run_id}, "
                f"Arxiv: {message.arxiv_enabled})")

    use_arxiv = message.arxiv_enabled

    async def generate():
        """
        The inner generator. Now much flatter and easier to read.
        Each section has one job.
        """
        # =====================================================================
        # SECTION 1: Setup Session
        # =====================================================================
        session_id = message.session_id or str(uuid.uuid4())
        logger.debug(f"[STREAM] Session ID: {session_id}")
        
        # Helper handles the try/except internally - non-fatal if it fails
        ensure_db_session(session_id, message.query)
        
        # Tell the frontend what session we're using
        yield sse_pack({'type': 'session', 'session_id': session_id})

        # =====================================================================
        # SECTION 2: Prepare Query
        # =====================================================================
        enhanced_query = prepare_enhanced_query(message.query, message.run_id)
        logger.debug(f"[STREAM] Final query: '{enhanced_query[:100]}...'")

        # =====================================================================
        # SECTION 3: Initialize Engine with Router
        # =====================================================================
        try:
            engine = get_arxiv_engine()
            logger.debug(f"[STREAM] Using ArxivAugmentedEngine (arxiv_enabled={use_arxiv})")
        except Exception as e:
            logger.error(f"[STREAM] Failed to initialize ArxivAugmentedEngine: {e}")
            yield sse_error(f"Failed to initialize engine: {e}")
            return

        # =====================================================================
        # SECTION 4: Stream Response with Router
        # =====================================================================
        chunk_count = 0
        try:
            # ArxivAugmentedEngine handles routing internally
            stream = engine.query_stream(
                enhanced_query, session_id,
                arxiv_enabled=use_arxiv, original_query=message.query
            )
            
            async for chunk in stream:
                chunk_count += 1
                if chunk_count % 10 == 0:
                    logger.debug(f"[STREAM] Chunk #{chunk_count}: {chunk.get('type', 'unknown')}")
                yield sse_pack(chunk)
            
            logger.info(f"[STREAM] Completed - {chunk_count} chunks streamed")

        except Exception as e:
            logger.error(f"[STREAM] Streaming error: {e}", exc_info=True)
            yield sse_error(f"Streaming error: {e}")
            return

        # =====================================================================
        # SECTION 5: Validation (Post-Stream Check)
        # =====================================================================
        if chunk_count == 0:
            logger.warning("[STREAM] No chunks generated - empty response")
            yield sse_error("No response generated from the query engine")

    # Return the streaming response with proper SSE headers
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
    if get_query_engine() is None:
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    try:
        history = get_arxiv_engine().get_history(session_id)
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
    if get_query_engine() is None:
        raise HTTPException(
            status_code=500, detail="QueryEngine not initialized properly"
        )

    try:
        get_arxiv_engine().clear_history(session_id)
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
    query_engine = get_query_engine()
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


@router.get("/knowledge-base")
async def list_knowledge_base_documents() -> Dict[str, Any]:
    """
    List all documents in the knowledge base.
    
    Returns distinct sources (arxiv papers and uploaded documents) with metadata.
    
    Returns:
        Dict with list of documents and their sources
    """
    from federated_pneumonia_detection.src.boundary.engine import get_engine
    from sqlalchemy import text
    
    try:
        engine = get_engine()
        
        # Query distinct sources from the langchain_pg_embedding table
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT 
                    cmetadata->>'source' as source,
                    cmetadata->>'paper_id' as paper_id,
                    COUNT(*) as chunk_count
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = 'research_papers'
                )
                GROUP BY cmetadata->>'source', cmetadata->>'paper_id'
                ORDER BY source
            """))
            
            documents = []
            for row in result:
                source = row[0] or "Unknown"
                paper_id = row[1]
                chunk_count = row[2]
                
                # Determine document type
                if source.startswith("arxiv:"):
                    doc_type = "arxiv"
                    display_name = source
                else:
                    doc_type = "uploaded"
                    display_name = source.split("/")[-1] if "/" in source else source
                
                documents.append({
                    "source": source,
                    "paper_id": paper_id,
                    "display_name": display_name,
                    "type": doc_type,
                    "chunk_count": chunk_count,
                })
        
        return {
            "documents": documents,
            "total_count": len(documents),
        }
        
    except Exception as e:
        logger.error(f"Error listing knowledge base: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error listing knowledge base: {str(e)}"
        )

