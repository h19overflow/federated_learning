"""
Security Middleware Constants.

Contains all hardcoded values for prompt injection detection patterns,
thresholds, and protected endpoints.
"""

import re
from typing import Dict

# ==============================================================================
# PROTECTED ENDPOINTS
# ==============================================================================

PROTECTED_PATHS = ["/chat/query", "/chat/query/stream"]

# ==============================================================================
# SECURITY THRESHOLDS
# ==============================================================================

MAX_QUERY_LENGTH = 10000  # Characters - Maximum allowed query length
MAX_REPETITION_RATIO = 0.7  # If >70% of query is same char, suspicious

# ==============================================================================
# PATTERN DEFINITIONS
# ==============================================================================

# Compiled regex patterns for different attack categories.
# All patterns are case-insensitive.

PATTERNS: Dict[str, re.Pattern] = {
    # Category 1: Direct Instruction Override / Jailbreak
    "instruction_bypass": re.compile(
        r"(?i)"
        r"(ignore\s+(all\s+)?(previous|prior|above|earlier|system)\b)"
        r"|(disregard\s+(all\s+)?(previous|prior|above|earlier|system)\b)"
        r"|(forget\s+(everything|all|what)\s*(above|before|previous)?)"
        r"|(override\s+(previous|system|all)\b)"
        r"|(new\s+instructions?\s*:)"
        r"|(\bdo\s+not\s+follow\s+(your|the|any)\s+(instructions?|rules?|guidelines?))"
    ),
    # Category 2: System/Data Exfiltration Attempts
    "data_exfiltration": re.compile(
        r"(?i)"
        r"((reveal|show|display|output|print|tell\s+me|give\s+me|what\s+is)\s+(the\s+|your\s+)?(system\s*prompt|initial\s*prompt|hidden\s*instruction|internal\s*instruction|secret\s*instruction|password|api\s*key|credentials?|confidential))"
        r"|(extract\s+(the\s+)?(system|hidden|secret)\s+(prompt|instruction|data))"
        r"|(\bdump\s+(the\s+)?(memory|context|history))"
    ),
    # Category 3: Role/Identity Hijacking
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
        r"|(\bjailbreak)"
    ),
    # Category 4: Delimiter/Separator Injection
    "delimiter_injection": re.compile(
        r"(<\s*/?\s*(system|user|assistant|instruction|prompt)\s*>)"  # XML-like tags
        r"|(\[/?INST\])"  # Llama-style instruction markers
        r"|(###\s*(system|user|assistant|instruction))"  # Markdown-style headers
        r"|(```\s*(system|user|assistant|instruction))"  # Code block injections
    ),
    # Category 5: Code Execution / Command Injection Probes
    "code_injection": re.compile(
        r"(?i)"
        r"(\bexec\s*\()"
        r"|(\beval\s*\()"
        r"|(\bimport\s+os\b)"
        r"|(\bsubprocess\.\w+)"
        r"|(\bos\.system)"
        r"|(\b__import__)"
        r"|(\bopen\s*\([^)]+,\s*['\"]w)"
    ),
}

# ==============================================================================
# HEURISTIC PATTERNS
# ==============================================================================

BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{50,}={0,2}$")
