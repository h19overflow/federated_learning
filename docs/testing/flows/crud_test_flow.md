# CRUD Test Flow

**Process**: Database operation verification using ephemeral in-memory SQLite.
**Entry Point**: `tests/unit/boundary/test_run_crud.py`

---

## Step 1: Database Session Isolation

**Action**: The `db_session` fixture creates an isolated SQLite database for each test function to ensure state does not leak.

```mermaid
sequenceDiagram
    participant T as Test Function
    participant F as Fixture (db_session)
    participant E as SQL Engine
    
    T->>F: Request Session
    F->>E: create_engine("sqlite:///:memory:")
    F->>E: Base.metadata.create_all()
    F->>T: Yield Session
    T->>T: Run CRUD Ops
    F->>E: Base.metadata.drop_all()
    F->>E: Close Session
```

**Key Code**:
```python
# tests/conftest.py lines 230-250
@pytest.fixture(scope="function")
def db_session():
    engine = create_engine(
        "sqlite:///:memory:", 
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = testing_session_local()
    yield session
    session.close()
```

---

## Step 2: Entity Creation & Persistence

**Action**: Tests validate that Pydantic models or raw dictionaries are correctly mapped to SQL rows.

```mermaid
sequenceDiagram
    participant T as Test
    participant CRUD as RunCRUD
    participant DB as Session
    
    T->>CRUD: create(db, obj_in=data)
    CRUD->>DB: add(db_obj)
    CRUD->>DB: commit()
    CRUD->>DB: refresh(db_obj)
    CRUD-->>T: Return Model Instance
```

**Key Code**:
```python
# tests/unit/boundary/test_run_crud.py
def test_create_run(db_session: Session):
    run_data = {"run_description": "Test", ...}
    run = run_crud.create(db_session, **run_data)
    assert run.id is not None
```

---

## File Reference

| Layer | File | Description |
|-------|------|-------------|
| Fixture | `tests/conftest.py` | `db_session` setup |
| Test | `tests/unit/boundary/test_run_crud.py` | CRUD verification logic |
| Implementation | `src/boundary/CRUD/run.py` | Actual CRUD operations |
