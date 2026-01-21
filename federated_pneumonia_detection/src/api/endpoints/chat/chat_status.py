"""Chat status endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from federated_pneumonia_detection.src.api.deps import get_mcp_manager

router = APIRouter()


@router.get("/arxiv/status")
async def get_arxiv_status() -> Dict[str, Any]:
    """Check if arxiv MCP server is available."""
    mcp_manager = get_mcp_manager()
    return {
        "available": mcp_manager.is_available,
        "tools": (
            [t.name for t in mcp_manager.get_arxiv_tools()]
            if mcp_manager.is_available
            else []
        ),
    }
