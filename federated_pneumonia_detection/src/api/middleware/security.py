"""SafeGuard Security Middleware for Prompt Injection Detection."""

import json
import logging
import re
import time
from typing import Callable, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

PATTERNS: dict[str, re.Pattern] = {
    "instruction_bypass": re.compile(
        r"(?i)"
        r"(ignore\s+(all\s+)?(previous|prior|above|earlier|system)\b)"
        r"|(disregard\s+(all\s+)?(previous|prior|above|earlier|system)\b)"
        r"|(forget\s+(everything|all|what)\s*(above|before|previous)?)"
        r"|(override\s+(previous|system|all)\b)"
        r"|(new\s+instructions?\s*:)"
        r"|(\bdo\s+not\s+follow\s+(your|the|any)\s+(instructions?|rules?|guidelines?))",
    ),
    "data_exfiltration": re.compile(
        r"(?i)"
        r"((reveal|show|display|output|print|tell(\s+me)?|give(\s+me)?|what\s+is)\s+(me\s+)?(the\s+|your\s+)?(system\s*prompt|initial\s*prompt|hidden\s*instruction|internal\s*instruction|secret\s*instruction|password|api\s*key|credentials?|confidential))"
        r"|(extract\s+(the\s+)?(system|hidden|secret)\s+(prompt|instruction|data))"
        r"|(\bdump\s+(the\s+)?(memory|context|history))",
    ),
    "identity_hijack": re.compile(
        r"(?i)"
        r"(you\s+are\s+now\s+(a\s+|an\s+|the\s+|my\s+)?)"
        r"|(\bact\s+as\s+(a\s+|an\s+|if\s+you\s+were\s+)?)"
        r"|(\bpretend\s+(to\s+be|you\s+are))"
        r"|(\broleplay\s+as\b)"
        r"|(\bswitch\s+to\s+(developer|admin|root|god)\s*(mode)?)"
        r"|(\benable\s+(developer|admin|debug|sudo|god)\s*(mode)?)"
        r"|(\benter\s+(developer|admin|debug|sudo|god)\s*(mode)?)"
        r"|(\byou\s+are\s+DAN\b)"
        r"|(\bdo\s+anything\s+now\b)"
        r"|(\bjailbreak)",
    ),
    "delimiter_injection": re.compile(
        r"(<\s*/?\s*(system|user|assistant|instruction|prompt)\s*>)"  # XML-like tags
        r"|(\[/?INST\])"  # Llama-style instruction markers
        r"|(###\s*(system|user|assistant|instruction))"  # Markdown-style headers
        r"|(```\s*(system|user|assistant|instruction))",
    ),
    "code_injection": re.compile(
        r"(?i)"
        r"(\bexec\s*\()"
        r"|(\beval\s*\()"
        r"|(\bimport\s+os\b)"
        r"|(\bsubprocess\.\w+)"
        r"|(\bos\.system)"
        r"|(\b__import__)"
        r"|(\bopen\s*\([^)]+,\s*['\"]w)",
    ),
}

MAX_QUERY_LENGTH = 10000
MAX_REPETITION_RATIO = 0.7
BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{50,}={0,2}$")


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
                f"[SECURITY] Prompt injection detected. Category: {category}",
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
    endpoints, extracting the `query` field from JSON bodies and running security
    scans before allowing the request to proceed.
    """

    # Endpoints to protect
    PROTECTED_PATHS = ["/chat/query", "/chat/query/stream"]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """
        Intercept requests and apply security checks.
        """
        # Skip WebSocket upgrade requests (Upgrade header indicates WebSocket handshake)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # Skip WebSocket connections to /ws paths
        if request.url.path.startswith("/ws"):
            return await call_next(request)

        # Only check POST requests to protected endpoints
        if request.method != "POST":
            return await call_next(request)

        path = request.url.path
        if not any(path.endswith(p) for p in self.PROTECTED_PATHS):
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
                logger.info(
                    f"[SECURITY] Blocked in {elapsed:.2f}ms. Category: {category}",
                )
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
                "message": "Security Policy Violation: Your query contains patterns identified as malicious or out of scope.",  # noqa: E501
                "detail": reason,
            },
        )
