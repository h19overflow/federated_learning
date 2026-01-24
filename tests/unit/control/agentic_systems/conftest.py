"""
Shared fixtures for agentic systems tests.

Provides mocks for LangChain components (LLMs, embeddings, vector stores),
database connections, and agentic system components.
"""

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.streaming import (
    SSEEventType,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.contracts import (
    AgentEvent,
    ChatInput,
)

# =============================================================================
# Mock LangChain Components
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create mock LangChain LLM."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="Mock LLM response")
    llm.invoke = Mock(return_value="Mock LLM response")
    llm.astream = AsyncMock()
    llm.stream = Mock()

    # Mock streaming chunks
    async def mock_astream_yield(*args, **kwargs):
        chunks = ["Hello ", "world ", "from ", "mock ", "LLM"]
        for chunk in chunks:
            mock_chunk = MagicMock()
            mock_chunk.content = chunk
            yield mock_chunk

    llm.astream.side_effect = mock_astream_yield
    llm.stream.side_effect = mock_astream_yield

    return llm


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    embeddings = MagicMock()
    embeddings.embed_documents = Mock(return_value=[[0.1] * 384] * 5)
    embeddings.embed_query = Mock(return_value=[0.1] * 384)
    return embeddings


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = MagicMock()

    # Mock retrieval
    sample_docs = [
        Document(
            page_content="Sample document content 1",
            metadata={"source": "paper1.pdf", "page": 1},
        ),
        Document(
            page_content="Sample document content 2",
            metadata={"source": "paper2.pdf", "page": 3},
        ),
    ]

    store.as_retriever = Mock(return_value=store)
    store.invoke = Mock(return_value=sample_docs)
    store.add_documents = Mock(return_value=["doc1_id", "doc2_id"])
    store.similarity_search = Mock(return_value=sample_docs)
    store.similarity_search_with_score = Mock(
        return_value=[(sample_docs[0], 0.9), (sample_docs[1], 0.8)],
    )

    return store


@pytest.fixture
def mock_retriever():
    """Create mock retriever."""
    retriever = MagicMock()

    sample_docs = [
        Document(
            page_content="Retrieved content 1",
            metadata={"source": "source1.pdf"},
        ),
        Document(
            page_content="Retrieved content 2",
            metadata={"source": "source2.pdf"},
        ),
    ]

    retriever.invoke = Mock(return_value=sample_docs)
    retriever.get_relevant_documents = Mock(return_value=sample_docs)

    return retriever


@pytest.fixture
def mock_bm25_retriever():
    """Create mock BM25 retriever."""
    retriever = MagicMock()
    sample_docs = [
        Document(page_content="BM25 result 1"),
        Document(page_content="BM25 result 2"),
    ]
    retriever.invoke = Mock(return_value=sample_docs)
    retriever.get_relevant_documents = Mock(return_value=sample_docs)
    retriever.k = 10
    return retriever


@pytest.fixture
def mock_ensemble_retriever(mock_bm25_retriever, mock_retriever):
    """Create mock ensemble retriever."""
    retriever = MagicMock()
    retriever.retrievers = [mock_bm25_retriever, mock_retriever]
    retriever.weights = [0.5, 0.5]

    sample_docs = [
        Document(page_content="Ensemble result 1"),
        Document(page_content="Ensemble result 2"),
    ]
    retriever.invoke = Mock(return_value=sample_docs)
    retriever.get_relevant_documents = Mock(return_value=sample_docs)

    return retriever


# =============================================================================
# Mock Database Components
# =============================================================================


@pytest.fixture
def mock_postgres_connection():
    """Create mock PostgreSQL connection."""
    conn = MagicMock()
    conn.cursor = MagicMock()
    conn.commit = Mock()
    conn.rollback = Mock()
    conn.close = Mock()
    return conn


@pytest.fixture
def mock_postgres_history():
    """Create mock PostgresChatMessageHistory."""
    history = MagicMock()

    # Mock messages
    history.messages = [
        HumanMessage(content="User question"),
        AIMessage(content="AI response"),
        HumanMessage(content="Another question"),
        AIMessage(content="Another response"),
    ]

    # Mock methods
    history.add_messages = Mock()
    history.clear = Mock()
    history.create_tables = Mock()

    return history


@pytest.fixture
def mock_history_manager(mock_postgres_history):
    """Create mock ChatHistoryManager."""
    manager = MagicMock()
    manager._get_postgres_history = Mock(return_value=mock_postgres_history)
    manager.add_to_history = Mock()
    manager.get_history = Mock(return_value=[("User", "AI"), ("User2", "AI2")])
    manager.clear_history = Mock()
    manager.format_for_context = Mock(
        return_value="User: User\nAI: AI\n\nUser2: User2\nAI: AI2",
    )
    manager.get_messages = Mock(return_value=mock_postgres_history.messages)
    return manager


@pytest.fixture
def mock_chat_session_crud():
    """Create mock chat session CRUD functions."""
    with (
        patch(
            "federated_pneumonia_detection.src.boundary.CRUD.chat_history.create_chat_session",
        ) as mock_create,
        patch(
            "federated_pneumonia_detection.src.boundary.CRUD.chat_history.get_chat_session",
        ) as mock_get,
        patch(
            "federated_pneumonia_detection.src.boundary.CRUD.chat_history.delete_chat_session",
        ) as mock_delete,
        patch(
            "federated_pneumonia_detection.src.boundary.CRUD.chat_history.get_all_chat_sessions",
        ) as mock_get_all,
    ):
        mock_create.return_value = MagicMock(
            session_id="test_session_id",
            title="Test Session",
        )
        mock_get.return_value = MagicMock(
            session_id="test_session_id",
            title="Test Session",
        )
        mock_delete.return_value = True
        mock_get_all.return_value = []

        yield {
            "create": mock_create,
            "get": mock_get,
            "delete": mock_delete,
            "get_all": mock_get_all,
        }


# =============================================================================
# Mock Tools
# =============================================================================


@pytest.fixture
def mock_base_tool():
    """Create mock LangChain BaseTool."""
    tool = MagicMock(spec=BaseTool)
    tool.name = "mock_tool"
    tool.description = "Mock tool for testing"
    tool.ainvoke = AsyncMock(return_value="Tool result")
    tool.invoke = Mock(return_value="Tool result")
    tool._run = Mock(return_value="Tool result")
    tool._arun = AsyncMock(return_value="Tool result")

    # Mock schema
    from pydantic import BaseModel

    class MockInput(BaseModel):
        query: str

    tool.args_schema = MockInput

    return tool


@pytest.fixture
def mock_rag_tool(mock_retriever):
    """Create mock RAG tool."""
    tool = MagicMock(spec=BaseTool)
    tool.name = "rag_search"
    tool.description = "Search knowledge base"
    tool.ainvoke = AsyncMock(
        return_value="Based on retrieved documents: federated learning enables privacy-preserving ML",
    )
    tool.invoke = Mock(
        return_value="Based on retrieved documents: federated learning enables privacy-preserving ML",
    )

    # Mock args schema
    from pydantic import BaseModel

    class RAGInput(BaseModel):
        query: str

    tool.args_schema = RAGInput

    return tool


@pytest.fixture
def mock_arxiv_tool():
    """Create mock ArXiv search tool."""
    tool = MagicMock(spec=BaseTool)
    tool.name = "arxiv_search"
    tool.description = "Search ArXiv papers"
    tool.ainvoke = AsyncMock(
        return_value="Found 5 papers on federated learning: Paper 1, Paper 2, ...",
    )
    tool.invoke = Mock(return_value="Found 5 papers on federated learning")

    # Mock args schema
    from pydantic import BaseModel

    class ArxivInput(BaseModel):
        query: str
        max_results: int = 10

    tool.args_schema = ArxivInput

    return tool


@pytest.fixture
def mock_tools_list(mock_rag_tool, mock_arxiv_tool):
    """Create list of mock tools."""
    return [mock_rag_tool, mock_arxiv_tool]


# =============================================================================
# Mock Agent Components
# =============================================================================


@pytest.fixture
def sample_chat_input():
    """Create sample ChatInput."""
    return ChatInput(
        query="What is federated learning?",
        session_id="test_session_123",
        arxiv_enabled=True,
        run_id=1,
    )


@pytest.fixture
def sample_agent_events():
    """Create sample agent events."""
    return [
        AgentEvent(type="session", session_id="test_session_123"),
        AgentEvent(type="status", content="Searching knowledge base..."),
        AgentEvent(type="token", content="Federated "),
        AgentEvent(type="token", content="learning "),
        AgentEvent(type="token", content="is a "),
        AgentEvent(type="token", content="distributed "),
        AgentEvent(type="token", content="approach."),
        AgentEvent(type="done", session_id="test_session_123"),
    ]


@pytest.fixture
def mock_base_agent():
    """Create mock BaseAgent."""
    agent = MagicMock()

    # Mock async stream
    async def mock_stream(chat_input: ChatInput) -> AsyncGenerator[AgentEvent, None]:
        yield AgentEvent(type="session", session_id=chat_input.session_id)
        yield AgentEvent(type="token", content="Mock ")
        yield AgentEvent(type="token", content="response")
        yield AgentEvent(type="done", session_id=chat_input.session_id)

    agent.stream = AsyncMock(side_effect=mock_stream)
    agent.query = AsyncMock(
        return_value={"answer": "Mock response", "session_id": "test"},
    )
    agent.history = Mock(return_value=[("User", "AI")])
    agent.clear_history = Mock()

    return agent


@pytest.fixture
def mock_arxiv_engine():
    """Create mock ArxivAugmentedEngine."""
    engine = MagicMock()

    async def mock_stream(chat_input: ChatInput) -> AsyncGenerator[AgentEvent, None]:
        session_id = chat_input.session_id if chat_input else "test_session"
        yield AgentEvent(type="session", session_id=session_id)
        yield AgentEvent(type="status", content="Searching...")
        yield AgentEvent(type="token", content="Research ")
        yield AgentEvent(type="token", content="results")
        yield AgentEvent(type="done", session_id=session_id)

    engine.stream = AsyncMock(side_effect=mock_stream)
    engine.query = AsyncMock(
        return_value={
            "answer": "Research findings...",
            "session_id": "test_session",
            "tools_used": [],
        },
    )
    engine.add_to_history = Mock()
    engine.get_history = Mock(return_value=[])
    engine.clear_history = Mock()

    return engine


# =============================================================================
# Mock Query Router
# =============================================================================


@pytest.fixture
def mock_query_router():
    """Create mock query router classification."""
    with patch(
        "federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.router._get_router_llm",
    ) as mock_get_llm:
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.mode = "research"
        mock_llm.invoke = Mock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        yield mock_llm


# =============================================================================
# Mock MCP Manager
# =============================================================================


@pytest.fixture
def mock_mcp_manager():
    """Create mock MCP Manager."""
    manager = MagicMock()
    manager._is_available = False
    manager._tools = []
    manager._client = None

    manager.is_available = False
    manager.get_arxiv_tools = Mock(return_value=[])
    manager.initialize = AsyncMock()
    manager.shutdown = AsyncMock()

    return manager


# =============================================================================
# Mock Query Engine (RAG)
# =============================================================================


@pytest.fixture
def mock_query_engine(
    mock_llm,
    mock_ensemble_retriever,
    mock_history_manager,
):
    """Create mock QueryEngine."""
    engine = MagicMock()
    engine.llm = mock_llm
    engine.ensemble_retriever = mock_ensemble_retriever
    engine.vector_store_retriever = mock_ensemble_retriever
    engine.bm25_retriever = mock_bm25_retriever()
    engine.history_manager = mock_history_manager

    engine.add_to_history = Mock()
    engine.get_history = Mock(return_value=[])
    engine.clear_history = Mock()
    engine.format_for_context = Mock(return_value="")

    engine.get_prompts = Mock(return_value=MagicMock())
    engine.get_chain = Mock(return_value=MagicMock())

    # Mock query_with_history
    async def mock_query_with_history_stream(*args, **kwargs):
        yield AgentEvent(type="token", content="Query ")
        yield AgentEvent(type="token", content="response")
        yield AgentEvent(type="done", session_id="test")

    engine.query_with_history_stream = AsyncMock(
        side_effect=mock_query_with_history_stream,
    )
    engine.query_with_history = Mock(
        return_value={
            "input": "test query",
            "context": [],
            "answer": "Test answer",
        },
    )

    return engine


# =============================================================================
# Async Test Helpers
# =============================================================================


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_stream_collector():
    """Helper to collect all items from an async generator."""

    async def collect(async_gen):
        items = []
        async for item in async_gen:
            items.append(item)
        return items

    return collect


# =============================================================================
# SSE Event Helpers
# =============================================================================


@pytest.fixture
def sse_event_factory():
    """Factory for creating SSE events."""

    def create_event(event_type: SSEEventType, **kwargs):
        from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.streaming import (
            create_sse_event,
        )

        return create_sse_event(event_type, **kwargs)

    return create_event


# =============================================================================
# Document Fixtures
# =============================================================================


@pytest.fixture
def sample_documents():
    """Create sample documents for RAG testing."""
    return [
        Document(
            page_content="Federated learning is a machine learning approach that enables training on decentralized data.",
            metadata={
                "source": "paper1.pdf",
                "page": 1,
                "title": "Introduction to Federated Learning",
            },
        ),
        Document(
            page_content="Differential privacy adds noise to gradients to protect individual data points.",
            metadata={"source": "paper2.pdf", "page": 3, "title": "Privacy in FL"},
        ),
        Document(
            page_content="Cross-silo federated learning involves multiple institutions training together.",
            metadata={"source": "paper3.pdf", "page": 5, "title": "Cross-Silo FL"},
        ),
    ]


@pytest.fixture
def sample_pdf_bytes():
    """Create sample PDF bytes for testing."""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test PDF) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000214 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n307\n%%EOF"


# =============================================================================
# Environment Mocks
# =============================================================================


@pytest.fixture(autouse=True)
def mock_agentic_env_vars(monkeypatch):
    """Mock environment variables for agentic systems tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test_db")
    monkeypatch.setenv("TESTING", "1")
    yield
