"""Content processing utilities for LLM responses.

Handles Gemini's list-based content format and response chunking.
"""

from __future__ import annotations

import logging
from typing import Any, Generator, List, Union

logger = logging.getLogger(__name__)


def normalize_content(content: Union[str, List[Any], None]) -> str:
    """
    Normalize LLM response content to a string.

    Gemini models may return content as:
    - A string (direct text)
    - A list of parts (each can be str or dict with 'text' key)
    - None

    Args:
        content: Raw content from LLM response

    Returns:
        Normalized string content

    Examples:
        >>> normalize_content("Hello world")
        "Hello world"

        >>> normalize_content([{"text": "Hello"}, " world"])
        "Hello world"

        >>> normalize_content(None)
        ""
    """
    if content is None:
        logger.debug("[Content] normalize_content received None")
        return ""

    if isinstance(content, str):
        logger.debug(f"[Content] normalize_content received string, length: {len(content)}")
        return content

    if isinstance(content, list):
        logger.debug(f"[Content] normalize_content received list with {len(content)} parts")
        parts = []
        for i, part in enumerate(content):
            if isinstance(part, str):
                parts.append(part)
                logger.debug(f"[Content] Part {i}: string, length {len(part)}")
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
                logger.debug(f"[Content] Part {i}: dict with text, length {len(part['text'])}")
            else:
                logger.warning(f"[Content] Part {i}: unexpected type {type(part)}")
        result = "".join(parts)
        logger.debug(f"[Content] Normalized list to string, final length: {len(result)}")
        return result

    # Fallback for unexpected types
    logger.warning(f"[Content] Unexpected content type: {type(content)}, using str()")
    return str(content)


def chunk_content(content: str, chunk_size: int = 50) -> Generator[str, None, None]:
    """
    Split content into chunks for simulated streaming.
    
    Used when we have a complete response but want to simulate
    token-by-token streaming for better UX.
    
    Args:
        content: Complete content string to chunk
        chunk_size: Maximum characters per chunk (default: 50)
    
    Yields:
        String chunks of at most chunk_size characters
    
    Examples:
        >>> list(chunk_content("Hello World", chunk_size=5))
        ["Hello", " Worl", "d"]
    """
    for i in range(0, len(content), chunk_size):
        yield content[i:i + chunk_size]
