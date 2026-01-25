"""Content processing utilities for LLM responses.

Handles Gemini's list-based content format and response chunking.
"""

from __future__ import annotations

import logging
from typing import Any, Generator, List, Union

logger = logging.getLogger(__name__)


def normalize_content(content: Union[str, List[Any], None]) -> str:
    """
    Normalize Gemini LLM response content to a string.

    Handles string, list of parts (str or dict with 'text' key), or None.

    Args:
        content: Raw LLM response content

    Returns:
        Normalized string content (empty string for None)
    """
    if content is None:
        logger.debug("[Content] normalize_content received None")
        return ""

    if isinstance(content, str):
        logger.debug(
            f"[Content] normalize_content received string, length: {len(content)}",
        )
        return content

    if isinstance(content, list):
        logger.debug(
            f"[Content] normalize_content received list with {len(content)} parts",
        )
        parts = []
        for i, part in enumerate(content):
            if isinstance(part, str):
                parts.append(part)
                logger.debug(f"[Content] Part {i}: string, length {len(part)}")
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
                logger.debug(
                    f"[Content] Part {i}: dict with text, length {len(part['text'])}",
                )
            else:
                logger.warning(f"[Content] Part {i}: unexpected type {type(part)}")
                parts.append(str(part))
        result = "".join(parts)
        logger.debug(
            f"[Content] Normalized list to string, final length: {len(result)}",
        )
        return result

    # Fallback for unexpected types
    logger.warning(f"[Content] Unexpected content type: {type(content)}, using str()")
    return str(content)


def chunk_content(content: str, chunk_size: int = 50) -> Generator[str, None, None]:
    """
    Split content into chunks for simulated streaming.

    Args:
        content: Complete content string to chunk
        chunk_size: Maximum characters per chunk (default: 50)

    Yields:
        String chunks of at most chunk_size characters
    """
    for i in range(0, len(content), chunk_size):
        yield content[i : i + chunk_size]
