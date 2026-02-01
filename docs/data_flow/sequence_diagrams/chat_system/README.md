# Chat System Documentation

**Purpose**: Comprehensive technical documentation for the research assistant chat system, covering streaming responses, session management, history retrieval, and arxiv integration.

---

## Table of Contents

1. **[Chat Streaming Flow](01_chat_stream_flow.md)**
   - SSE token-by-token response delivery
   - Session initialization and query enhancement
   - Agent factory and chat agent setup
   - Stream execution with tool integration (RAG + Arxiv)
   - Real-time frontend updates

2. **[Session Management Flow](02_session_management_flow.md)**
   - List all chat sessions
   - Create new session with auto-generated titles
   - Delete session and clear history
   - Auto-creation during chat (ensure_session)
   - Singleton pattern implementation

3. **[Chat History Flow](03_chat_history_flow.md)**
   - Paginated history retrieval
   - Agent factory initialization
   - History persistence and storage
   - BaseAgent contract implementation

4. **[Arxiv Status Flow](04_arxiv_status_flow.md)**
   - MCP server availability check
   - 5-minute cache with TTL
   - Tool discovery
   - Frontend integration patterns

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    React Frontend (SSE Client)                │
│  - EventSource for streaming                                 │
│  - Session management UI                                     │
│  - History pagination                                        │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP/SSE
                         ↓
┌──────────────────────────────────────────────────────────────┐
│              FastAPI Chat Endpoints (/api/chat)               │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ chat_stream  │  │ chat_sessions│  │ chat_history │      │
│  │ (SSE)        │  │ (CRUD)       │  │ (pagination) │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┴──────────────────┘               │
│                            │                                  │
└────────────────────────────┼──────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                  Control Layer (Agents)                       │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           AgentFactory (Singleton)                   │    │
│  │  - Creates and caches ArxivAugmentedEngine          │    │
│  │  - Uses app.state for pre-initialized services      │    │
│  └────────────────────┬────────────────────────────────┘    │
│                       ↓                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │      ArxivAugmentedEngine (BaseAgent)               │    │
│  │  - ChatGoogleGenerativeAI (Gemini 2.5 Flash)        │    │
│  │  - RAG Tool (QueryEngine)                           │    │
│  │  - Arxiv MCP Tools (search, metadata)               │    │
│  │  - ChatHistoryManager (PostgreSQL)                  │    │
│  └────────────────────┬────────────────────────────────┘    │
│                       │                                      │
└───────────────────────┼──────────────────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────────────────┐
│                Data Layer (PostgreSQL)                        │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ ChatSession  │  │ ChatHistory  │                         │
│  │              │  │              │                         │
│  │ - id (UUID)  │  │ - session_id │                         │
│  │ - title      │  │ - user_msg   │                         │
│  │ - timestamps │  │ - ai_response│                         │
│  └──────────────┘  └──────────────┘                         │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Component Quick Reference

### API Endpoints

| Endpoint | Method | Purpose | File |
|----------|--------|---------|------|
| `/chat/query/stream` | POST | Stream chat response (SSE) | `chat_stream.py:34` |
| `/chat/sessions` | GET | List all chat sessions | `chat_sessions.py:24` |
| `/chat/sessions` | POST | Create new session | `chat_sessions.py:31` |
| `/chat/sessions/{id}` | DELETE | Delete session + history | `chat_sessions.py:43` |
| `/chat/history/{id}` | GET | Get paginated history | `chat_history.py:19` |
| `/chat/arxiv/status` | GET | Check MCP availability | `chat_status.py:21` |

### Core Components

| Component | Purpose | File | Pattern |
|-----------|---------|------|---------|
| **SessionManager** | Coordinates session persistence & history | `session_manager.py` | Singleton |
| **AgentFactory** | Creates and caches agents | `factory.py` | Singleton |
| **ArxivAugmentedEngine** | Research agent with RAG + Arxiv | `research_engine.py` | BaseAgent |
| **ChatHistoryManager** | PostgreSQL history persistence | `postgres_history.py` | Singleton |
| **MCPManager** | Manages arxiv MCP server connection | `mcp_manager.py` | Singleton |

---

## Data Flow Summary

### Chat Streaming Flow

```
User sends query
  ↓
POST /api/chat/query/stream {query, session_id?, run_id?, arxiv_enabled}
  ↓
SessionManager.ensure_session() → Create if not exists
  ↓
prepare_enhanced_query() → Fetch run context if run_id provided
  ↓
AgentFactory.get_chat_agent() → Return ArxivAugmentedEngine
  ↓
agent.stream(ChatInput) → Start async generator
  ↓
LLM astream_events() with tools (RAG + Arxiv)
  ↓
For each event:
  ├─ Token → yield {type: "token", content: "..."}
  ├─ Tool call → yield {type: "tool_call", tool: "...", args: {...}}
  └─ Done → yield {type: "done"}
  ↓
sse_pack() → Format as "data: {...}\n\n"
  ↓
StreamingResponse → SSE to frontend
  ↓
ChatHistoryManager.add_to_history() → Persist conversation turn
```

### Session Management Flow

```
Frontend needs session list
  ↓
GET /api/chat/sessions
  ↓
SessionManager.list_sessions() → get_all_chat_sessions()
  ↓
Query: SELECT * FROM chat_sessions ORDER BY updated_at DESC
  ↓
Convert to schema: [{id, title, created_at, updated_at}, ...]
  ↓
Return to frontend
```

### History Retrieval Flow

```
Frontend requests history
  ↓
GET /api/chat/history/{session_id}?limit=50&offset=0
  ↓
AgentFactory.get_chat_agent() → ArxivAugmentedEngine
  ↓
agent.history(session_id) → ChatHistoryManager.get_history()
  ↓
Query: SELECT * FROM chat_history WHERE session_id = ... ORDER BY created_at ASC
  ↓
Convert to tuples: [(user_msg, ai_response), ...]
  ↓
Apply pagination: history[offset:offset+limit]
  ↓
Format: [{"user": "...", "assistant": "..."}, ...]
  ↓
Return: {history, session_id, total_count, limit, offset}
```

---

## SSE Event Types

| Event Type | Payload | Frontend Action |
|------------|---------|-----------------|
| `session` | `{type: "session", session_id: "..."}` | Store session_id |
| `token` | `{type: "token", content: "word"}` | Append to UI |
| `tool_call` | `{type: "tool_call", tool: "...", args: {...}}` | Show tool badge |
| `status` | `{type: "status", message: "..."}` | Update status |
| `done` | `{type: "done"}` | Mark complete |
| `error` | `{type: "error", message: "..."}` | Display error |

---

## Database Schema

### ChatSession Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key (auto or provided) |
| `title` | String | Session title (user or auto-generated) |
| `created_at` | DateTime | Session creation timestamp |
| `updated_at` | DateTime | Last activity timestamp |

### ChatHistory Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `session_id` | UUID | FK to ChatSession |
| `user_message` | Text | User's query |
| `ai_response` | Text | Agent's response |
| `created_at` | DateTime | Message timestamp |

**Indexes**:
- `chat_sessions.updated_at` (for sorting)
- `chat_history.session_id` (for filtering)
- `chat_history.created_at` (for ordering)

---

## Key Design Patterns

| Pattern | Implementation | Benefit |
|---------|----------------|---------|
| **Singleton** | SessionManager, AgentFactory, MCPManager | Single source of truth, resource efficiency |
| **Factory** | AgentFactory.get_chat_agent() | Lazy initialization, dependency injection |
| **Server-Sent Events** | StreamingResponse + async generator | Real-time updates, low latency |
| **Pagination** | limit/offset params | Memory efficiency, progressive loading |
| **Caching** | 5-min TTL for arxiv status | Reduced MCP queries, faster responses |
| **Dependency Injection** | app.state for pre-initialized services | Startup optimization, testability |
| **BaseAgent Contract** | Abstract methods for stream, query, history | Interface segregation, polymorphism |

---

## Run Context Enhancement

When `run_id` is provided in chat query, the system enhances the query with training run context:

**Example Input**:
```json
{
  "query": "What was the best recall?",
  "session_id": "abc-123",
  "run_id": 456,
  "arxiv_enabled": false
}
```

**Enhanced Query** (sent to LLM):
```
What was the best recall?

[TRAINING RUN CONTEXT - Run #456]
============================================================
Training Mode: federated
Status: completed
Duration: 1:15:00

METRICS SUMMARY (450 total metrics recorded):
------------------------------------------------------------
validation_recall:
  - Best: 0.9200
  - Worst: 0.7500
  - Average: 0.8600

FEDERATED LEARNING DETAILS:
------------------------------------------------------------
Number of Clients: 5
Server Evaluations: 10 rounds
...
```

**Files**:
- `chat_utils.py:274-320` (enhance_query_with_run_context)
- `chat_utils.py:53-244` (build_run_context)

---

## Agent Tools Integration

### RAG Tool
**Purpose**: Search internal training documentation
**Implementation**: `create_rag_tool(query_engine)`
**Triggered When**: User asks about training results, model architecture

### Arxiv Tools (MCP)
**Purpose**: Search academic papers on arxiv
**Implementation**: MCP server via stdio transport
**Triggered When**: `arxiv_enabled=true` and user asks research questions

**Example Tool Calls**:
```json
// RAG search
{
  "type": "tool_call",
  "tool": "rag_search",
  "args": {"query": "pneumonia detection metrics"}
}

// Arxiv search
{
  "type": "tool_call",
  "tool": "search_arxiv",
  "args": {"query": "federated learning medical imaging"}
}
```

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| **List sessions** | 10-50 ms | Database query |
| **Create session** | 20-100 ms | DB insert + title generation |
| **Delete session** | 30-150 ms | DB delete + history cascade |
| **Stream first token** | 500-2000 ms | LLM cold start + context |
| **Stream subsequent tokens** | 20-100 ms | LLM streaming |
| **History retrieval** | 50-500 ms | Depends on total messages |
| **Arxiv status (cached)** | < 1 ms | Memory access |
| **Arxiv status (uncached)** | 50-200 ms | MCP round-trip |

---

## Error Handling Strategies

| Layer | Strategy | Example |
|-------|----------|---------|
| **API** | HTTPException with detail | 404 Session not found, 500 Agent init failed |
| **Agent** | Graceful degradation | RAG unavailable → Arxiv-only mode |
| **Session** | Non-fatal logging | Session creation fails → Log warning, continue |
| **History** | Empty result | No history → Return empty list |
| **MCP** | Availability flag | MCP down → `available=false`, disable features |

---

## Frontend Integration Patterns

### EventSource (SSE Client)
```typescript
const eventSource = new EventSource('/api/chat/query/stream', {
  method: 'POST',
  body: JSON.stringify({query, session_id, run_id, arxiv_enabled})
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'session':
      setSessionId(data.session_id);
      break;
    case 'token':
      appendToken(data.content);
      break;
    case 'tool_call':
      showToolIndicator(data.tool);
      break;
    case 'done':
      markComplete();
      break;
  }
};
```

### Session Management
```typescript
// Load sessions on mount
useEffect(() => {
  fetch('/api/chat/sessions')
    .then(res => res.json())
    .then(sessions => setSessions(sessions));
}, []);

// Create new session
const createSession = async (initialQuery) => {
  const response = await fetch('/api/chat/sessions', {
    method: 'POST',
    body: JSON.stringify({initial_query: initialQuery})
  });
  const session = await response.json();
  return session.id;
};

// Delete session
const deleteSession = async (sessionId) => {
  await fetch(`/api/chat/sessions/${sessionId}`, {method: 'DELETE'});
  setSessions(prev => prev.filter(s => s.id !== sessionId));
};
```

### History Pagination
```typescript
const [history, setHistory] = useState([]);
const [hasMore, setHasMore] = useState(true);
const [page, setPage] = useState(0);

const loadHistory = async (sessionId, pageNum) => {
  const offset = pageNum * 50;
  const response = await fetch(
    `/api/chat/history/${sessionId}?limit=50&offset=${offset}`
  );
  const data = await response.json();

  setHistory(prev => [...prev, ...data.history]);
  setHasMore(offset + data.history.length < data.total_count);
  setPage(pageNum);
};

// Infinite scroll
const handleScroll = () => {
  if (hasMore && !loading && scrolledToTop) {
    loadHistory(sessionId, page + 1);
  }
};
```

---

## Testing Strategies

### Unit Tests
```python
# Test session manager singleton
def test_session_manager_singleton():
    mgr1 = SessionManager.get_instance()
    mgr2 = SessionManager.get_instance()
    assert mgr1 is mgr2

# Test cache expiration
def test_arxiv_status_cache_expiration(monkeypatch):
    # Mock time to simulate TTL expiration
    ...
```

### Integration Tests
```python
# Test full stream flow
async def test_chat_stream_flow(client):
    response = client.post('/api/chat/query/stream', json={
        'query': 'test',
        'session_id': 'test-123',
        'arxiv_enabled': False
    })
    assert response.headers['content-type'] == 'text/event-stream'

    events = []
    for line in response.iter_lines():
        if line.startswith('data:'):
            events.append(json.loads(line[6:]))

    assert events[0]['type'] == 'session'
    assert any(e['type'] == 'done' for e in events)
```

---

## Monitoring & Observability

### Key Metrics
- **Stream latency**: Time to first token, total response time
- **Session count**: Active sessions, sessions created/deleted per day
- **History size**: Messages per session (avg, max)
- **MCP availability**: Uptime percentage, connection errors
- **Cache hit rate**: Arxiv status cache effectiveness

### Logging Points
```python
# chat_stream.py
logger.info("[STREAM] Query: '%s...', Session: %s, Run: %s", ...)

# session_manager.py
logger.info("[SessionManager] Generated title: '%s'", title)
logger.warning("[SessionManager] Failed to ensure DB session (non-fatal): %s", exc)

# chat_history.py
logger.warning("[HISTORY] No cached factory in app.state, creating on-demand")
logger.error("Error retrieving history: %s", exc)
```

---

## Security Considerations

| Concern | Mitigation |
|---------|------------|
| **SQL Injection** | SQLAlchemy ORM, parameterized queries |
| **Session hijacking** | UUID session IDs (unpredictable) |
| **XSS in chat** | Frontend escapes HTML in messages |
| **Prompt injection** | Middleware checks malicious prompts |
| **Rate limiting** | API rate limits (configured in middleware) |
| **CORS** | Configured allowed origins |

---

## File Organization

```
docs/data_flow/sequence_diagrams/chat_system/
├── README.md (this file)
├── 01_chat_stream_flow.md
├── 02_session_management_flow.md
├── 03_chat_history_flow.md
└── 04_arxiv_status_flow.md

federated_pneumonia_detection/src/
├── api/endpoints/chat/
│   ├── chat_endpoints.py (router assembly)
│   ├── chat_stream.py (SSE streaming)
│   ├── chat_sessions.py (CRUD operations)
│   ├── chat_history.py (paginated retrieval)
│   ├── chat_status.py (arxiv availability)
│   └── chat_utils.py (context enhancement)
│
└── control/agentic_systems/multi_agent_systems/
    ├── __init__.py (exports)
    ├── contracts.py (AgentEvent, ChatInput, AgentEventType)
    ├── base_agent.py (BaseAgent interface)
    ├── factory.py (AgentFactory)
    ├── session_manager.py (SessionManager)
    └── chat/
        ├── agents/research_engine.py (ArxivAugmentedEngine)
        ├── history/postgres_history.py (ChatHistoryManager)
        └── providers/
            ├── rag.py (QueryEngine)
            ├── titles.py (generate_chat_title)
            └── tools/
                └── rag_tool.py (create_rag_tool)
```

---

## Related Documentation

- **API Layer CLAUDE.md**: `federated_pneumonia_detection/src/api/CLAUDE.md`
- **Control Layer CLAUDE.md**: `federated_pneumonia_detection/src/control/CLAUDE.md`
- **Metrics Collection Docs**: `docs/data_flow/sequence_diagrams/metrics_collection/`

---

**Last Updated**: 2026-02-01
**Documentation Pattern**: tech-doc-principal (step-based, file references, mermaid diagrams)
