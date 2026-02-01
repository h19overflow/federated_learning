# Chat Session Management Flow

**API Base**: `/api/chat/sessions`
**Files**: `chat_sessions.py`, `session_manager.py`
**Pattern**: Singleton SessionManager + Database Persistence

---

## Step 1: List All Sessions

**API**: `GET /api/chat/sessions`
**Entry Point**: `chat_sessions.py:24` → `list_chat_sessions()`

```mermaid
sequenceDiagram
    participant Frontend as React Frontend
    participant API as chat_sessions.py
    participant SessionMgr as SessionManager
    participant CRUD as chat_history CRUD
    participant DB as Database

    Frontend->>API: GET /api/chat/sessions

    API->>SessionMgr: list_sessions()
    Note right of SessionMgr: session_manager.py:54-56

    SessionMgr->>CRUD: get_all_chat_sessions()
    Note right of CRUD: boundary/CRUD/chat_history.py

    CRUD->>DB: SELECT * FROM chat_sessions ORDER BY updated_at DESC

    DB-->>CRUD: List[ChatSession]

    CRUD-->>SessionMgr: sessions

    SessionMgr-->>API: sessions

    API->>API: [_to_schema(session) for session in sessions]
    Note right of API: lines 28<br/>Convert to API schema

    loop For each session
        API->>API: _to_schema(session)
        Note right of API: lines 53-60

        API->>API: ChatSessionSchema(id, title, created_at, updated_at)
    end

    API-->>Frontend: List[ChatSessionSchema]
    Note right of Frontend: [{id, title, created_at, updated_at}, ...]
```

**Key Code**:
```python
# chat_sessions.py lines 24-28
@router.get("/sessions", response_model=List[ChatSessionSchema])
async def list_chat_sessions() -> List[ChatSessionSchema]:
    """List all available chat sessions."""
    sessions = session_manager.list_sessions()
    return [_to_schema(session) for session in sessions]
```

```python
# chat_sessions.py lines 53-60
def _to_schema(session: ChatSession) -> ChatSessionSchema:
    """Convert a ChatSession model to API schema."""
    return ChatSessionSchema(
        id=str(session.id),
        title=str(session.title),
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )
```

---

## Step 2: Create New Session

**API**: `POST /api/chat/sessions`
**Entry Point**: `chat_sessions.py:31` → `create_new_chat_session()`

```mermaid
sequenceDiagram
    participant Frontend as React Frontend
    participant API as chat_sessions.py
    participant SessionMgr as SessionManager
    participant TitleGen as generate_chat_title
    participant CRUD as chat_history CRUD
    participant DB as Database

    Frontend->>API: POST /api/chat/sessions
    Note right of Frontend: {title?: "...", initial_query?: "..."}

    API->>SessionMgr: create_session(title, initial_query)
    Note right of SessionMgr: session_manager.py:58-67

    alt title not provided AND initial_query provided
        SessionMgr->>TitleGen: generate_chat_title(initial_query)
        Note right of TitleGen: providers/titles.py<br/>LLM-based title generation

        TitleGen->>TitleGen: Extract key phrases from query
        TitleGen-->>SessionMgr: generated_title (e.g., "Pneumonia Detection Analysis")

        SessionMgr->>SessionMgr: title = generated_title
        Note right of SessionMgr: line 65-66
    else title provided
        SessionMgr->>SessionMgr: Use provided title
    end

    SessionMgr->>CRUD: create_chat_session(title=title)
    Note right of CRUD: boundary/CRUD/chat_history.py

    CRUD->>DB: INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (uuid, title, NOW(), NOW())

    DB-->>CRUD: ChatSession

    CRUD-->>SessionMgr: session

    SessionMgr-->>API: session

    API->>API: _to_schema(session)
    Note right of API: line 40

    API-->>Frontend: ChatSessionSchema
    Note right of Frontend: {id, title, created_at, updated_at}
```

**Key Code**:
```python
# chat_sessions.py lines 31-40
@router.post("/sessions", response_model=ChatSessionSchema)
async def create_new_chat_session(
    request: CreateSessionRequest = CreateSessionRequest(),
) -> ChatSessionSchema:
    """Create a new chat session with optional auto-generated title."""
    session = session_manager.create_session(
        title=request.title,
        initial_query=request.initial_query,
    )
    return _to_schema(session)
```

```python
# session_manager.py lines 58-67
def create_session(
    self,
    title: Optional[str] = None,
    initial_query: Optional[str] = None,
) -> ChatSession:
    """Create a new chat session with optional title generation."""
    if not title and initial_query:
        title = generate_chat_title(initial_query)
        logger.info("[SessionManager] Generated title: '%s'", title)
    return create_chat_session(title=title)
```

---

## Step 3: Delete Session

**API**: `DELETE /api/chat/sessions/{session_id}`
**Entry Point**: `chat_sessions.py:43` → `delete_existing_chat_session()`

```mermaid
sequenceDiagram
    participant Frontend as React Frontend
    participant API as chat_sessions.py
    participant SessionMgr as SessionManager
    participant CRUD as chat_history CRUD
    participant HistoryMgr as ChatHistoryManager
    participant DB as Database

    Frontend->>API: DELETE /api/chat/sessions/abc-123

    API->>SessionMgr: delete_session(session_id)
    Note right of SessionMgr: session_manager.py:81-83

    SessionMgr->>CRUD: delete_chat_session(session_id)
    Note right of CRUD: boundary/CRUD/chat_history.py

    CRUD->>DB: DELETE FROM chat_sessions WHERE id = 'abc-123'

    alt Session found and deleted
        DB-->>CRUD: 1 row affected
        CRUD-->>SessionMgr: True
    else Session not found
        DB-->>CRUD: 0 rows affected
        CRUD-->>SessionMgr: False
    end

    alt success == False
        API->>Frontend: HTTPException(404, "Session not found")
        Note right of API: line 48
    end

    API->>SessionMgr: clear_history(session_id)
    Note right of SessionMgr: session_manager.py:85-87

    SessionMgr->>HistoryMgr: clear_history(session_id)
    Note right of HistoryMgr: chat/history/postgres_history.py

    HistoryMgr->>DB: DELETE FROM chat_history WHERE session_id = 'abc-123'

    DB-->>HistoryMgr: History cleared

    HistoryMgr-->>SessionMgr: Cleared

    SessionMgr-->>API: Cleared

    API-->>Frontend: {message: "Session abc-123 deleted"}
    Note right of API: line 50
```

**Key Code**:
```python
# chat_sessions.py lines 43-50
@router.delete("/sessions/{session_id}")
async def delete_existing_chat_session(session_id: str) -> dict:
    """Delete a chat session and clear its history."""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    session_manager.clear_history(session_id)
    return {"message": f"Session {session_id} deleted"}
```

```python
# session_manager.py lines 81-87
def delete_session(self, session_id: str) -> bool:
    """Delete a chat session."""
    return delete_chat_session(session_id)

def clear_history(self, session_id: str) -> None:
    """Clear conversation history for a session."""
    self._history_manager.clear_history(session_id)
```

---

## Step 4: Ensure Session (Auto-Creation During Chat)

**Triggered By**: `chat_stream.py:48` during streaming
**Entry Point**: `session_manager.py:69` → `ensure_session()`

```mermaid
sequenceDiagram
    participant Stream as chat_stream.py
    participant SessionMgr as SessionManager
    participant CRUD as chat_history CRUD
    participant DB as Database

    Note over Stream: User sends chat message<br/>with session_id

    Stream->>SessionMgr: ensure_session(session_id, query)
    Note right of SessionMgr: session_manager.py:69-79

    SessionMgr->>CRUD: get_chat_session(session_id)

    CRUD->>DB: SELECT * FROM chat_sessions WHERE id = session_id

    alt Session exists
        DB-->>CRUD: ChatSession
        CRUD-->>SessionMgr: session
        Note right of SessionMgr: Session already exists, nothing to do
    else Session doesn't exist
        DB-->>CRUD: None
        CRUD-->>SessionMgr: None

        SessionMgr->>CRUD: create_chat_session(title=query[:50]..., session_id=session_id)
        Note right of SessionMgr: line 74

        CRUD->>DB: INSERT INTO chat_sessions (id, title, ...) VALUES (session_id, ...)

        DB-->>CRUD: New session
        CRUD-->>SessionMgr: session
    end

    SessionMgr-->>Stream: Session ensured
```

**Key Code**:
```python
# session_manager.py lines 69-79
def ensure_session(self, session_id: str, query: str) -> None:
    """Ensure a database-backed session exists for a session id."""
    try:
        existing = get_chat_session(session_id)
        if not existing:
            create_chat_session(title=f"{query[:50]}...", session_id=session_id)
    except Exception as exc:
        logger.warning(
            "[SessionManager] Failed to ensure DB session (non-fatal): %s",
            exc,
        )
```

---

## Singleton Pattern Implementation

**File**: `session_manager.py` (lines 26-52)

```mermaid
sequenceDiagram
    participant Thread1 as Thread 1
    participant Thread2 as Thread 2
    participant SessionMgr as SessionManager
    participant Lock as Threading Lock

    Thread1->>SessionMgr: get_instance()
    Note right of SessionMgr: lines 47-52

    alt _instance is None
        SessionMgr->>Lock: Acquire lock
        Note right of Lock: lines 34

        alt _instance still None (double-check)
            SessionMgr->>SessionMgr: Create new instance
            Note right of SessionMgr: __new__() lines 32-38

            SessionMgr->>SessionMgr: __init__()
            Note right of SessionMgr: lines 40-45

            SessionMgr->>SessionMgr: self._history_manager = ChatHistoryManager()
            SessionMgr->>SessionMgr: self._initialized = True
        end

        Lock->>Lock: Release lock
    end

    SessionMgr-->>Thread1: SessionManager instance

    Thread2->>SessionMgr: get_instance()
    Note right of SessionMgr: _instance already exists

    SessionMgr-->>Thread2: Same SessionManager instance
```

**Key Code**:
```python
# session_manager.py lines 26-52
class SessionManager:
    """Coordinates chat session persistence and history cleanup (Singleton)."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        # Only initialize once
        if self._initialized:
            return
        self._history_manager = ChatHistoryManager()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "SessionManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls()
        return cls._instance
```

---

## File Reference

| Component | File | Key Lines |
|-----------|------|-----------|
| **API Endpoints** | `chat_sessions.py` | 24-60 |
| **Session Manager** | `session_manager.py` | 26-100 |
| **CRUD Operations** | `boundary/CRUD/chat_history.py` | (create, get, delete, list) |
| **Title Generation** | `chat/providers/titles.py` | (generate_chat_title) |
| **History Manager** | `chat/history/postgres_history.py` | (ChatHistoryManager) |

---

## Database Schema

### ChatSession Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key (auto-generated or provided) |
| `title` | String | Session title (user-provided or auto-generated) |
| `created_at` | DateTime | Session creation timestamp |
| `updated_at` | DateTime | Last activity timestamp |

### ChatHistory Table (Related)
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `session_id` | UUID | FK to ChatSession |
| `user_message` | Text | User's query |
| `ai_response` | Text | Agent's response |
| `created_at` | DateTime | Message timestamp |

---

## API Request/Response Examples

### List Sessions
```json
// GET /api/chat/sessions
Response: [
  {
    "id": "abc-123-def-456",
    "title": "Federated Learning Results Analysis",
    "created_at": "2026-02-01T10:30:00Z",
    "updated_at": "2026-02-01T11:45:00Z"
  },
  {
    "id": "xyz-789-uvw-012",
    "title": "Model Performance Comparison",
    "created_at": "2026-01-31T14:20:00Z",
    "updated_at": "2026-01-31T15:30:00Z"
  }
]
```

### Create Session
```json
// POST /api/chat/sessions
Request: {
  "initial_query": "What was the best recall in my last training run?"
}

Response: {
  "id": "new-session-uuid",
  "title": "Training Run Recall Analysis",  // Auto-generated
  "created_at": "2026-02-01T12:00:00Z",
  "updated_at": "2026-02-01T12:00:00Z"
}
```

### Delete Session
```json
// DELETE /api/chat/sessions/abc-123

Response: {
  "message": "Session abc-123 deleted"
}

// Error case (404)
{
  "detail": "Session not found"
}
```

---

## Session Lifecycle

```
1. User opens chat interface
   ↓
2. Frontend checks for existing session_id in localStorage
   ↓
   ├─ If exists: Use existing session_id
   └─ If not: Generate new UUID, don't create session yet
   ↓
3. User sends first message
   ↓
4. POST /api/chat/query/stream with session_id
   ↓
5. Backend: ensure_session() auto-creates if not exists
   ↓
6. Session persisted in database
   ↓
7. Subsequent messages use same session_id
   ↓
8. User deletes session
   ↓
9. DELETE /api/chat/sessions/{id}
   ↓
10. Session + history removed from database
```

---

## Title Generation Logic

**File**: `chat/providers/titles.py`

```python
# Simplified example
def generate_chat_title(query: str) -> str:
    """Generate a concise title from the query using LLM."""
    # Uses LLM to extract key phrases
    # Falls back to query[:50]... if LLM fails

    prompt = f"Generate a concise 3-5 word title for this query: {query}"
    # Call LLM, return title
    return title or f"{query[:50]}..."
```

**Examples**:
- Query: "What was the best recall in my federated training run with 5 clients?"
  - Title: "Federated Training Recall Analysis"

- Query: "Compare precision between centralized and federated modes"
  - Title: "Precision Mode Comparison"

---

## Error Handling

| Error Scenario | Handler | Response |
|----------------|---------|----------|
| Session not found (delete) | lines 47-48 | HTTPException(404, "Session not found") |
| Session creation fails (ensure) | lines 75-79 | Log warning (non-fatal, continue) |
| Title generation fails | titles.py | Fallback to query[:50]... |
| History clear fails | Non-blocking | Log error, continue |

---

## Concurrency Handling

**Pattern**: Thread-safe singleton with lock

```python
# Double-check locking pattern
if cls._instance is None:
    with cls._lock:  # Acquire lock
        if cls._instance is None:  # Double-check
            cls._instance = super().__new__(cls)
```

**Why**: Ensures only one SessionManager instance exists across all threads/requests, preventing:
- Multiple ChatHistoryManager instances
- Inconsistent session state
- Memory leaks from duplicate instances
