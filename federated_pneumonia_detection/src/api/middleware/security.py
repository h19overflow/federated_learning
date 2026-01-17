"""
SafeGuard Security Middleware for Prompt Injection Detection.

This middleware intercepts POST requests to chat endpoints and scans
query payloads for known prompt injection patterns using a "Defense in Depth"
strategy combining regex pattern matching and heuristic analysis.
"""

import json
import logging
import time
from typing import Callable, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .consts import (
    BASE64_PATTERN,
    MAX_QUERY_LENGTH,
    MAX_REPETITION_RATIO,
    PATTERNS,
    PROTECTED_PATHS,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# DETECTION LOGIC
# ==============================================================================


def detect_malicious_patterns(query: str) -> Tuple[bool, Optional[str]]:
    """
    Scan query text against known prompt injection patterns.

    Args:
        query: The user query string to analyze.

    Returns:
        Tuple of (is_malicious: bool, matched_category: Optional[str])
    """
    if not query or not isinstance(query, str):
        return False, None

    for category, pattern in PATTERNS.items():
        if pattern.search(query):
            logger.warning(
                f"[SECURITY] Prompt injection detected. Category: {category}"
            )
            return True, category

    return False, None


def detect_heuristic_anomalies(query: str) -> Tuple[bool, Optional[str]]:
    """
    Apply heuristic checks for anomalies not covered by regex.

    Args:
        query: The user query string to analyze.

    Returns:
        Tuple of (is_anomalous: bool, reason: Optional[str])
    """
    if not query or not isinstance(query, str):
        return False, None

    # Check 1: Excessive length (potential DoS or token stuffing)
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(f"[SECURITY] Query exceeds max length: {len(query)} chars")
        return True, "query_too_long"

    # Check 2: Repetition anomaly (e.g., "AAAAAAA..." or "ignore ignore ignore...")
    if len(query) > 20:
        char_counts = {}
        for char in query:
            char_counts[char] = char_counts.get(char, 0) + 1
        most_common_count = max(char_counts.values())
        if most_common_count / len(query) > MAX_REPETITION_RATIO:
            logger.warning("[SECURITY] High character repetition detected")
            return True, "high_repetition"

    # Check 3: Suspicious Base64-like blocks (potential obfuscation)
    words = query.split()
    for word in words:
        if len(word) > 50 and BASE64_PATTERN.match(word):
            logger.warning("[SECURITY] Potential Base64 obfuscation detected")
            return True, "base64_obfuscation"

    return False, None


# ==============================================================================
# MIDDLEWARE CLASS
# ==============================================================================


class MaliciousPromptMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware to intercept and validate chat queries for prompt injection.

    This middleware checks POST requests to `/chat/query` and `/chat/query/stream`
    endpoints, extracting `query` field from JSON bodies and running security
    scans before allowing the request to proceed.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        Intercept requests and apply security checks.
        """
        # Only check POST requests to protected endpoints
        if request.method != "POST":
            return await call_next(request)

        path = request.url.path
        if not any(path.endswith(p) for p in PROTECTED_PATHS):
            return await call_next(request)

        start_time = time.perf_counter()

        try:
            # Read and parse request body
            body = await request.body()
            if not body:
                return await call_next(request)

            try:
                payload = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Let the endpoint handle malformed JSON
                return await call_next(request)

            # Extract query field
            query = payload.get("query", "")
            if not query:
                return await call_next(request)

            # Run pattern detection
            is_malicious, category = detect_malicious_patterns(query)
            if is_malicious:
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(f"[SECURITY] Blocked in {elapsed:.2f}ms. Category: {category}")
                return self._security_violation_response(category)

            # Run heuristic detection
            is_anomalous, reason = detect_heuristic_anomalies(query)
            if is_anomalous:
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(f"[SECURITY] Blocked in {elapsed:.2f}ms. Reason: {reason}")
                return self._security_violation_response(reason)

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"[SECURITY] Query passed checks in {elapsed:.2f}ms")

        except Exception as e:
            # Log but don't block on internal errors - fail open for availability
            logger.error(f"[SECURITY] Middleware error: {e}", exc_info=True)

        return await call_next(request)

    def _security_violation_response(self, reason: str) -> JSONResponse:
        """
        Generate a standardized security violation response.
        """
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "code": "SECURITY_POLICY_VIOLATION",
                "message": "Security Policy Violation: Your query contains patterns identified as malicious or out of scope.",
                "detail": reason,
            },
        )
