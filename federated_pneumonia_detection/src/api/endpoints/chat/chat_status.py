"""Chat status endpoints."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter

from federated_pneumonia_detection.src.api.deps import get_mcp_manager

router = APIRouter()

# Cache for arxiv status with TTL
_arxiv_status_cache: Dict[str, Any] | None = None
_arxiv_status_cache_time: float = 0
CACHE_TTL_SECONDS: int = 300  # 5 minutes


@router.get("/arxiv/status")
async def get_arxiv_status() -> Dict[str, Any]:
    """Check if arxiv MCP server is available (cached with 5-min TTL)."""
    global _arxiv_status_cache, _arxiv_status_cache_time

    current_time = time.time()

    # Check if cache is valid
    if (
        _arxiv_status_cache
        and (current_time - _arxiv_status_cache_time) < CACHE_TTL_SECONDS
    ):
        return _arxiv_status_cache

    # Cache miss or expired - fetch fresh data
    mcp_manager = get_mcp_manager()

    result = {
        "available": mcp_manager.is_available,
        "tools": (
            [t.name for t in mcp_manager.get_arxiv_tools()]
            if mcp_manager.is_available
            else []
        ),
        "cached_at": datetime.utcnow().isoformat(),
    }

    # Update cache
    _arxiv_status_cache = result
    _arxiv_status_cache_time = current_time

    return result
