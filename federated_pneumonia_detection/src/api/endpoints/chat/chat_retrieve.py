"""Chat retrieval and knowledge base endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from federated_pneumonia_detection.src.api.deps import get_query_engine
from federated_pneumonia_detection.src.api.endpoints.schema import ChatMessage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/retrieve")
async def retrieve_documents(
    message: ChatMessage,
    request: Request = None,
) -> Dict[str, Any]:
    """
    Retrieve documents based on a query without generating an answer.

    Args:
        message: Chat message with query
        request: FastAPI request object for accessing app.state

    Returns:
        Dictionary with retrieved documents
    """
    query_engine = get_query_engine(app_state=request.app.state if request else None)
    if query_engine is None:
        raise HTTPException(
            status_code=500,
            detail="QueryEngine not initialized properly",
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
                },
            )
        return {"documents": documents}
    except Exception as exc:
        logger.error("Error retrieving documents: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving documents: {str(exc)}",
        )


@router.get("/knowledge-base")
async def list_knowledge_base_documents() -> Dict[str, Any]:
    """List all documents in the knowledge base."""
    import asyncio
    from sqlalchemy import text

    from federated_pneumonia_detection.src.boundary.engine import get_engine

    def _fetch_documents_sync():
        """Synchronous DB operation to run in executor."""
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                SELECT DISTINCT
                    cmetadata->>'source' as source,
                    cmetadata->>'paper_id' as paper_id,
                    COUNT(*) as chunk_count,
                    CASE 
                        WHEN cmetadata->>'source' LIKE 'arxiv:%' THEN 'arxiv'
                        ELSE 'uploaded'
                    END as doc_type,
                    CASE 
                        WHEN cmetadata->>'source' LIKE 'arxiv:%' THEN cmetadata->>'source'
                        WHEN cmetadata->>'source' LIKE '%/%' THEN 
                            SUBSTRING(cmetadata->>'source' FROM '[^/]*$')
                        ELSE cmetadata->>'source'
                    END as display_name
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = 'research_papers'
                )
                GROUP BY cmetadata->>'source', cmetadata->>'paper_id'
                ORDER BY source
            """,
                ),
            )

            documents = []
            for row in result:
                source = row[0] or "Unknown"
                paper_id = row[1]
                chunk_count = row[2]
                doc_type = row[3]
                display_name = row[4]

                documents.append(
                    {
                        "source": source,
                        "paper_id": paper_id,
                        "display_name": display_name,
                        "type": doc_type,
                        "chunk_count": chunk_count,
                    },
                )

            return documents

    try:
        # Run blocking DB operation in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, _fetch_documents_sync)

        return {"documents": documents, "total_count": len(documents)}

    except Exception as exc:
        logger.error("Error listing knowledge base: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing knowledge base: {str(exc)}",
        )
