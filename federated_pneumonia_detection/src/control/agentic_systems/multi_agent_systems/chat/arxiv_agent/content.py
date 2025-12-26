"""Content processing utilities for LLM responses.

Handles Gemini's list-based content format and response chunking.
"""

from __future__ import annotations

from typing import Any, Generator, List, Union


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
        return ""
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    
    # Fallback for unexpected types
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
