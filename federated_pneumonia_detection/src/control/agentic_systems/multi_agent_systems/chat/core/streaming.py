"""SSE streaming utilities and event type definitions.

Provides standardized event creation for Server-Sent Events streaming.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class SSEEventType(str, Enum):
    """Event types for SSE streaming responses."""

    TOKEN = "token"  # nosec B105 - SSE event type, not a password
    STATUS = "status"  # Status update message
    TOOL_CALL = "tool_call"  # Tool execution notification
    ERROR = "error"  # Error message
    DONE = "done"  # Stream completion signal


def create_sse_event(
    event_type: SSEEventType,
    content: Optional[str] = None,
    message: Optional[str] = None,
    tool: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Create a standardized SSE event dictionary.

    Args:
        event_type: Type of SSE event (token, status, error, etc.)
        content: Content for token/status events
        message: Error message for error events
        tool: Tool name for tool_call events
        args: Tool arguments for tool_call events
        session_id: Session ID for done events
        **kwargs: Additional fields to include

    Returns:
        Dict with type and relevant fields based on event type

    Examples:
        >>> create_sse_event(SSEEventType.TOKEN, content="Hello")
        {"type": "token", "content": "Hello"}

        >>> create_sse_event(SSEEventType.ERROR, message="Failed")
        {"type": "error", "message": "Failed"}
    """
    event: Dict[str, Any] = {"type": event_type.value}

    if event_type == SSEEventType.TOKEN:
        event["content"] = content or ""
    elif event_type == SSEEventType.STATUS:
        event["content"] = content or ""
    elif event_type == SSEEventType.TOOL_CALL:
        event["tool"] = tool
        event["args"] = args or {}
    elif event_type == SSEEventType.ERROR:
        event["message"] = message or "Unknown error"
    elif event_type == SSEEventType.DONE:
        event["session_id"] = session_id

    # Add any additional kwargs
    event.update(kwargs)

    return event


async def execute_tool_async(
    tool_name: str,
    tool_args: Dict[str, Any],
    tools: list,
) -> tuple[Any, Optional[str]]:
    """
    Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments to pass to the tool
        tools: List of available tool objects

    Returns:
        Tuple of (result, error_message). Error is None on success.
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.debug(
        f"[ToolExec] Searching for tool '{tool_name}' among {len(tools)} available tools",
    )

    for tool in tools:
        if tool.name == tool_name:
            try:
                logger.debug(
                    f"[ToolExec] Found tool '{tool_name}'. Invoking with args: {tool_args}",
                )
                result = await tool.ainvoke(tool_args)
                result_preview = str(result)[:200] if result else "None"
                logger.info(
                    f"[ToolExec] Tool {tool_name} executed successfully. Result length: {len(str(result))}, Preview: {result_preview}",
                )
                return result, None
            except Exception as e:
                error_msg = f"Tool error: {str(e)}"
                logger.error(
                    f"[ToolExec] Tool {tool_name} failed with exception: {e}",
                    exc_info=True,
                )
                return error_msg, str(e)

    available_tools = [t.name for t in tools]
    logger.error(
        f"[ToolExec] Tool '{tool_name}' not found. Available tools: {available_tools}",
    )
    return None, f"Tool '{tool_name}' not found"
