# Logging and Exception Handling Improvements

**Date:** 2026-01-09
**Issue:** Generic error message "Sorry, I encountered an error while processing your query" displayed to user without specific error details in logs.

## Root Cause Analysis

From the logs:
```
INFO - [ArxivEngine] No content, attempting stream... - engine.py - 304
INFO - AFC is enabled with max remote calls: 10. - models.py - 7186
```

The stream attempted to generate a response after tool execution, but:
1. No logging showed what chunks were received during streaming
2. No logging before yielding the "No response after tool execution" error
3. Tool execution errors weren't logged with detailed context
4. Generic top-level exception handler masked the specific failure point

## Files Modified

### 1. `engine.py` - ArxivAugmentedEngine
**Location:** `C:\Users\User\Projects\FYP2\federated_pneumonia_detection\src\control\agentic_systems\multi_agent_systems\chat\arxiv_agent\engine.py`

#### Changes:

**Tool Execution (lines 256-282):**
- Added logging for tool arguments
- Added try-except around individual tool execution
- Log tool success with result length
- Log and handle tool errors explicitly before returning
- Return early with ERROR event if tool fails

**Tool Result Processing (lines 284-308):**
- Added try-except around message appending
- Log number of tool results being appended
- Debug log for each ToolMessage created
- Log model invocation results (has tool_calls, has content)
- Early return with ERROR event on failure

**Response Generation After Tools (lines 314-376):**
- Added comprehensive try-except around entire response generation
- Log response content length before processing
- Log normalized content length
- Track and log chunk counts during streaming
- Log first 3 chunks for debugging
- Log total chunks and response length after streaming
- Better error messages distinguishing different failure modes:
  - "Model generated empty response after normalization"
  - "Streaming error: {specific error}"
  - "No response generated after tool execution. Please try rephrasing your question."
  - "Response generation failed: {specific error}"

**Basic Mode (lines 229-250):**
- Added chunk counting
- Added try-except around streaming
- Log chunk count and response length
- Error handling with early return

**Direct Streaming (No Tools) (lines 378-399):**
- Added chunk counting
- Added try-except around streaming
- Log chunk count and response length
- Error handling with early return

**History Saving (lines 401-411):**
- Added try-except around history save
- Log session ID, query length, response length
- Non-fatal error handling (don't fail if history save fails)

### 2. `query_router.py` - Query Classification
**Location:** `C:\Users\User\Projects\FYP2\federated_pneumonia_detection\src\control\agentic_systems\multi_agent_systems\chat\arxiv_agent\query_router.py`

#### Changes (lines 70-95):
- Added debug log before model invocation
- Added debug log showing raw classification response
- Enhanced exception logging with `exc_info=True`
- Added info log when defaulting to research mode

### 3. `streaming.py` - Tool Execution Utility
**Location:** `C:\Users\User\Projects\FYP2\federated_pneumonia_detection\src\control\agentic_systems\multi_agent_systems\chat\arxiv_agent\streaming.py`

#### Changes (lines 73-109):
- Added debug log showing search for tool
- Added debug log when tool is found with args
- Log result preview (first 200 chars)
- Enhanced success logging with result length
- Enhanced error logging with `exc_info=True`
- Log available tools when tool not found

### 4. `content.py` - Content Normalization
**Location:** `C:\Users\User\Projects\FYP2\federated_pneumonia_detection\src\control\agentic_systems\multi_agent_systems\chat\arxiv_agent\content.py`

#### Changes (lines 14-65):
- Added logging import
- Debug log for None content
- Debug log for string content with length
- Debug log for list content with part count
- Debug log for each part in list (type and length)
- Warning log for unexpected part types
- Debug log for final normalized length
- Warning log for unexpected content types with fallback

## Logging Levels Used

- **DEBUG:** Detailed trace information (chunk contents, message counts, content parsing)
- **INFO:** Key operation milestones (tool execution success, stream completion, classification)
- **WARNING:** Unexpected but handled conditions (invalid classification, unexpected content types)
- **ERROR:** Failures requiring user attention (tool errors, streaming failures, empty responses)

All ERROR logs include `exc_info=True` for full stack traces.

## Error Event Flow

1. **Tool Execution Fails:** Immediate ERROR event + early return
2. **Message Processing Fails:** ERROR event + early return
3. **Model Invocation Fails:** ERROR event + early return
4. **Streaming Fails:** Caught in try-except, ERROR event + early return
5. **Empty Response:** Specific ERROR event with helpful message

Each error path:
- Logs with ERROR level including stack trace
- Yields SSE ERROR event to frontend
- Returns early to prevent further processing

## Expected Log Output (Success Path)

```
INFO - [QueryRouter] Classifying query: 'I want to research...'
DEBUG - [QueryRouter] Invoking classification model...
DEBUG - [QueryRouter] Raw response: 'research'
INFO - [QueryRouter] Classification: research
INFO - [ArxivEngine] Retrieved 1 tools: ['search_local_knowledge_base']
INFO - [ArxivEngine] Built 6 messages including history
INFO - [ArxivEngine] Invoking model to check for tool calls...
INFO - [ArxivEngine] Model requested 1 tool calls
INFO - [ArxivEngine] Executing tool: search_local_knowledge_base with args: {...}
DEBUG - [ToolExec] Searching for tool 'search_local_knowledge_base' among 1 available tools
DEBUG - [ToolExec] Found tool 'search_local_knowledge_base'. Invoking with args: {...}
INFO - [ToolExec] Tool search_local_knowledge_base executed successfully. Result length: 6374, Preview: ...
INFO - [ArxivEngine] Tool search_local_knowledge_base succeeded. Result length: 6374
INFO - [ArxivEngine] Appending 1 tool results to messages
DEBUG - [ArxivEngine] Added ToolMessage for search_local_knowledge_base, content length: 6374
INFO - [ArxivEngine] Checking for more tool calls...
INFO - [ArxivEngine] Model invoked. Has tool_calls: False, Has content: True
INFO - [ArxivEngine] No more tool calls. Generating final response.
INFO - [ArxivEngine] Response has content. Length: 847
DEBUG - [Content] normalize_content received string, length: 847
INFO - [ArxivEngine] Normalized content length: 847
INFO - [ArxivEngine] Yielded 17 chunks. Total response length: 847
INFO - [ArxivEngine] Saved to history. Session: ..., Query length: 95, Response length: 847
INFO - [ArxivEngine] Stream complete for session ...
```

## Expected Log Output (Failure Path - Empty Response)

```
INFO - [ArxivEngine] No content in response, attempting astream...
DEBUG - [ArxivEngine] Messages count before astream: 8
DEBUG - [ArxivEngine] Chunk 1: ...
DEBUG - [ArxivEngine] Chunk 2: ...
INFO - [ArxivEngine] Astream complete. Received 5 chunks, full_response length: 0
ERROR - [ArxivEngine] No response after tool execution. Chunks received: 5, Messages: 8
```

## Benefits

1. **Precise Error Location:** Logs show exactly which phase failed
2. **Context-Rich Errors:** Tool names, chunk counts, message counts, content lengths
3. **Graceful Degradation:** Early returns prevent cascading failures
4. **User-Friendly Messages:** Specific error messages instead of generic "error occurred"
5. **Debugging Clarity:** Debug logs for trace-level investigation
6. **Non-Fatal Handling:** History save failures don't crash the response

## Testing Recommendations

1. Test the original failing query to see detailed logs
2. Verify error messages are specific and helpful
3. Check that stack traces appear in logs for exceptions
4. Confirm frontend receives descriptive error events
5. Test all three modes: basic, research with tools, research without tools
