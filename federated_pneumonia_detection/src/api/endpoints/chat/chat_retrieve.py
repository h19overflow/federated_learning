"""Chat retrieval and knowledge base endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from federated_pneumonia_detection.src.api.deps import get_query_engine
from federated_pneumonia_detection.src.api.endpoints.schema import ChatMessage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/retrieve")
async def retrieve_documents(message: ChatMessage) -> Dict[str, Any]:
    """Retrieve documents based on a query without generating an answer."""
    query_engine = get_query_engine()
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
    from sqlalchemy import text

    from federated_pneumonia_detection.src.boundary.engine import get_engine

    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
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
            """,
                ),
            )

            documents = []
            for row in result:
                source = row[0] or "Unknown"
                paper_id = row[1]
                chunk_count = row[2]

                if source.startswith("arxiv:"):
                    doc_type = "arxiv"
                    display_name = source
                else:
                    doc_type = "uploaded"
                    display_name = source.split("/")[-1] if "/" in source else source

                documents.append(
                    {
                        "source": source,
                        "paper_id": paper_id,
                        "display_name": display_name,
                        "type": doc_type,
                        "chunk_count": chunk_count,
                    },
                )

        return {"documents": documents, "total_count": len(documents)}

    except Exception as exc:
        logger.error("Error listing knowledge base: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing knowledge base: {str(exc)}",
        )
