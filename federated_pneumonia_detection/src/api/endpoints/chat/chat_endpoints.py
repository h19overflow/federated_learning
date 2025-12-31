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


# ==============================================================================
# HELPER FUNCTIONS (Extracted to reduce nesting in endpoints)
# ==============================================================================

def sse_pack(data: dict) -> str:
    """
    Pack a dictionary into Server-Sent Events format.
    
    This standardizes the SSE output so we don't repeat the formatting everywhere.
    """
    return f"data: {json.dumps(data)}\n\n"


def sse_error(message: str, error_type: str = "error") -> str:
    """Create a formatted SSE error message."""
    return sse_pack({"type": error_type, "message": message})


def ensure_db_session(session_id: str, query: str) -> None:
    """
    Ensure a chat session exists in the database.
    
    This is a non-critical operation - if it fails, we log and continue.
    The chat will still work, just without persistent session tracking.
    """
    try:
        existing_session = get_chat_session(session_id)
        if not existing_session:
            logger.info(f"[Helper] Creating new DB session: {session_id}")
            create_chat_session(title=query[:50] + "...", session_id=session_id)
    except Exception as e:
        # Non-fatal: we can still chat even if DB session tracking fails
        logger.warning(f"[Helper] Failed to ensure DB session (non-fatal): {e}")


def prepare_enhanced_query(query: str, run_id: Optional[int]) -> str:
    """
    Enhance a query with run context if a run_id is provided.
    
    If enhancement fails, returns the original query (graceful degradation).
    """
    if run_id is None:
        return query
    
    try:
        logger.info(f"[Helper] Enhancing query with run context for run_id: {run_id}")
        enhanced = enhance_query_with_run_context(query, run_id)
        logger.info("[Helper] Query enhanced successfully")
        return enhanced
    except Exception as e:
        logger.error(f"[Helper] Failed to enhance query (using original): {e}")
        return query  # Graceful degradation

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
    logger.info(f"[STREAM] Query: '{message.query[:100]}...', "
                f"Session: {message.session_id}, Run: {message.run_id}, "
                f"Arxiv: {message.arxiv_enabled}")
    
    use_arxiv = message.arxiv_enabled

    # --- GUARD CLAUSE: Fail fast if no engine is available ---
    if not use_arxiv and query_engine is None:
        logger.error("[STREAM] QueryEngine not initialized")
        raise HTTPException(status_code=500, detail="QueryEngine not initialized properly")

    async def generate():
        """
        The inner generator. Now much flatter and easier to read.
        Each section has one job.
        """
        # =====================================================================
        # SECTION 1: Setup Session
        # =====================================================================
        session_id = message.session_id or str(uuid.uuid4())
        logger.info(f"[STREAM] Session ID: {session_id}")
        
        # Helper handles the try/except internally - non-fatal if it fails
        ensure_db_session(session_id, message.query)
        
        # Tell the frontend what session we're using
        yield sse_pack({'type': 'session', 'session_id': session_id})

        # =====================================================================
        # SECTION 2: Prepare Query
        # =====================================================================
        enhanced_query = prepare_enhanced_query(message.query, message.run_id)
        logger.info(f"[STREAM] Final query: '{enhanced_query[:100]}...'")

        # =====================================================================
        # SECTION 3: Choose Engine (Guard Clause Style)
        # =====================================================================
        if use_arxiv:
            try:
                engine = get_arxiv_engine()
                logger.info("[STREAM] Using ArxivAugmentedEngine")
            except Exception as e:
                logger.error(f"[STREAM] Failed to initialize ArxivAugmentedEngine: {e}")
                yield sse_error(f"Failed to initialize Arxiv engine: {e}")
                return  # Early exit - no nesting needed!
        else:
            engine = query_engine
            logger.info("[STREAM] Using standard QueryEngine")

        # =====================================================================
        # SECTION 4: Stream Response (The Main Event)
        # =====================================================================
        chunk_count = 0
        try:
            # Unified streaming logic - both engines expose similar interfaces
            if use_arxiv:
                stream = engine.query_stream(
                    enhanced_query, session_id, 
                    arxiv_enabled=True, original_query=message.query
                )
            else:
                stream = engine.query_with_history_stream(
                    enhanced_query, session_id, original_query=message.query
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


@router.get("/knowledge-base")
async def list_knowledge_base_documents() -> Dict[str, Any]:
    """
    List all documents in the knowledge base.
    
    Returns distinct sources (arxiv papers and uploaded documents) with metadata.
    
    Returns:
        Dict with list of documents and their sources
    """
    from federated_pneumonia_detection.src.boundary.engine import settings, get_engine
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

