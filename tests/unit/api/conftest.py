"""
Shared fixtures for API endpoint tests.

Enhances database mocking with proper dependency injection for:
- Database sessions (SQLAlchemy)
- Session management (chat sessions)
- Agent factory (chat agents with streaming)

Usage:
    Use api_client_with_all_mocks for complete mocking of all dependencies.
    Use individual fixtures (mock_db_session, mock_session_manager, etc.)
    for specific test scenarios.
Phase 1 Implementation - TEST_FAILURE_ANALYSIS_AND_FIX_PLAN.md
"""

import datetime
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# Set test mode before any imports
os.environ["TESTING"] = "1"
os.environ["USE_ALEMBIC"] = "false"

# Prevent database engine creation during imports
# Mock the engine module BEFORE any app imports
mock_engine = MagicMock()
sys.modules["federated_pneumonia_detection.src.boundary.engine"] = MagicMock(
    get_engine=lambda: mock_engine,
    create_tables=lambda: None,
    dispose_engine=lambda: None,
    _engine=mock_engine,
    SessionLocal=MagicMock(),
)

from federated_pneumonia_detection.src.api import deps  # noqa: E402
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (  # noqa: E402
    InferencePrediction,
    InferenceResponse,
)
from federated_pneumonia_detection.src.api.main import app  # noqa: E402


@pytest.fixture
def mock_db_session():
    """
    Mock database session that properly simulates SQLAlchemy behavior.

    Provides a MagicMock that mimics common SQLAlchemy session operations:
    - query().filter() chains
    - add(), commit(), rollback(), refresh(), close()
    - Proper return values for filter chains

    Returns:
        MagicMock: Mock session with SQLAlchemy-like behavior
    """
    mock_session = MagicMock(spec=Session)
    mock_session.query.return_value = MagicMock()
    mock_session.add.return_value = None
    mock_session.commit.return_value = None
    mock_session.rollback.return_value = None
    mock_session.refresh.return_value = None
    mock_session.close.return_value = None
    mock_session.flush.return_value = None
    mock_session.expunge.return_value = None

    # Mock common query patterns with proper chaining
    def mock_filter_method(*args, **kwargs):
        """Mock filter method that returns chainable query object."""
        result_mock = MagicMock()
        result_mock.first.return_value = None
        result_mock.all.return_value = []
        result_mock.filter.return_value = result_mock  # Chainable
        result_mock.order_by.return_value = result_mock  # Chainable
        result_mock.limit.return_value = result_mock  # Chainable
        result_mock.offset.return_value = result_mock  # Chainable
        result_mock.join.return_value = result_mock  # Chainable
        result_mock.options.return_value = result_mock  # Chainable
        return result_mock

    # Apply mock filter to query
    mock_session.query.return_value.filter = mock_filter_method
    mock_session.query.return_value.order_by = mock_filter_method
    mock_session.query.return_value.limit = mock_filter_method
    mock_session.query.return_value.offset = mock_filter_method
    mock_session.query.return_value.join = mock_filter_method
    mock_session.query.return_value.options = mock_filter_method

    return mock_session


@pytest.fixture
def api_client_with_db(mock_db_session, mock_inference_service):
    """
    FastAPI TestClient with database dependency properly mocked.

    Overrides the get_db() dependency to use the mock session instead
    of a real database connection. Use this for API tests that need
    database interaction but don't require chat/agent functionality.

    Args:
        mock_db_session: Mock database session fixture
        mock_inference_service: Mock inference service fixture

    Yields:
        TestClient: FastAPI test client with mocked database
    """

    def override_get_db():
        """Override function that yields mock session."""
        try:
            yield mock_db_session
        finally:
            pass

    def override_get_inference_service():
        return mock_inference_service

    def override_get_query_engine():
        return MagicMock()

    app.dependency_overrides[deps.get_db] = override_get_db
    app.dependency_overrides[deps.get_inference_service] = (
        override_get_inference_service
    )
    app.dependency_overrides[deps.get_query_engine] = override_get_query_engine

    # Mock startup services to prevent database connection during app startup
    with patch("federated_pneumonia_detection.src.api.main.initialize_services"):
        with TestClient(app) as client:
            yield client

    app.dependency_overrides.clear()


@pytest.fixture
def mock_inference_service(mock_inference_engine):
    """Mock InferenceService with sensible defaults."""
    from fastapi import HTTPException

    mock = MagicMock()

    # Mock validator to raise HTTPException for invalid files
    def mock_validate_or_raise(file):
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be PNG or JPEG.",
            )

    # Create a side effect function that calls the engine and builds a response
    async def predict_single_impl(file):
        # Validate first (will raise if invalid)
        mock_validate_or_raise(file)
        # Call the engine's predict method
        predicted_class, confidence, pneumonia_prob, normal_prob = (
            mock_inference_engine.predict(None)
        )
        return InferenceResponse(
            prediction=InferencePrediction(
                predicted_class=predicted_class,
                confidence=confidence,
                pneumonia_probability=pneumonia_prob,
                normal_probability=normal_prob,
            ),
            processing_time_ms=100.0,
            model_version="mock-1.0.0",
        )

    # Use AsyncMock with side_effect to call the engine
    mock.predict_single = AsyncMock(side_effect=predict_single_impl)

    mock.is_ready.return_value = True
    mock.check_ready_or_raise.return_value = None
    mock.get_info.return_value = {
        "status": "healthy",
        "model_loaded": True,
        "gpu_available": False,
        "model_version": "mock-1.0.0",
    }

    # Attach the engine mock to the service
    mock.engine = mock_inference_engine

    # Attach the validator with the side effect
    mock.validator.validate_or_raise.side_effect = mock_validate_or_raise
    return mock


@pytest.fixture
def mock_session_manager():
    """
    Mock SessionManager for chat endpoints.

    Simulates SessionManager singleton behavior for chat session operations:
    - create_session(): Creates new chat sessions
    - list_sessions(): Returns list of sessions
    - get_session(): Retrieves specific session
    - delete_session(): Removes a session
    - get_session_history(): Returns message history

    Returns:
        MagicMock: Mock session manager with predefined behaviors
    """
    mock = MagicMock()

    # Mock session object with typical attributes
    mock_session = MagicMock()
    mock_session.id = "test-session-id"
    mock_session.title = "Test Session"
    mock_session.messages = []
    mock_session.created_at = None
    mock_session.updated_at = None

    # Configure common method returns
    mock.create_session.return_value = mock_session
    mock.list_sessions.return_value = []
    mock.get_session.return_value = mock_session
    mock.delete_session.return_value = True
    mock.ensure_session.return_value = None
    mock.clear_history.return_value = None

    # Mock session history with sample messages
    mock.get_session_history.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    return mock


@pytest.fixture
def mock_agent_factory():
    """
    Mock AgentFactory for chat endpoints with streaming support.

    Simulates AgentFactory behavior for chat agent operations:
    - get_chat_agent(): Returns agent with stream() method
    - stream(): Async generator that yields chat events
    - history(): Returns conversation history

    The mock stream yields events in the expected format:
    - {"type": "session", "session_id": "test-id"}
    - {"type": "token", "content": "Hello"}
    - {"type": "done"}

    Returns:
        MagicMock: Mock agent factory with streaming chat agent
    """
    mock = MagicMock()

    # Mock streaming response with typical chat events
    async def mock_stream(*args, **kwargs):
        """Async generator that yields mock chat events."""
        yield {"type": "session", "session_id": "test-id"}
        yield {"type": "token", "content": "Hello"}
        yield {"type": "token", "content": " world"}
        yield {"type": "done"}

    # Mock agent with stream method
    mock_orchestrator = MagicMock()
    mock_orchestrator.stream = mock_stream
    mock_orchestrator.history.return_value = []
    mock_orchestrator.query.return_value = "Mock response"

    mock.get_chat_agent.return_value = mock_orchestrator
    return mock


@pytest.fixture
def api_client_with_all_mocks(
    mock_db_session, mock_session_manager, mock_agent_factory, mock_inference_service
):
    """
    Fully mocked TestClient for API tests.

    Combines all dependency overrides into a single fixture:
    - Database session (mock_db_session)
    - Session manager (mock_session_manager)
    - Agent factory (mock_agent_factory)
    - Inference service (mock_inference_service)

    This is the recommended fixture for comprehensive API endpoint testing
    as it eliminates external dependencies while maintaining realistic behavior.

    Args:
        mock_db_session: Mock database session fixture
        mock_session_manager: Mock session manager fixture
        mock_agent_factory: Mock agent factory fixture
        mock_inference_service: Mock inference service fixture

    Yields:
        TestClient: FastAPI test client with all dependencies mocked
    """
    from federated_pneumonia_detection.src.api.deps import get_mcp_manager

    def override_get_db():
        """Override function that yields mock database session."""
        yield mock_db_session

    def override_get_inference_service():
        return mock_inference_service

    def override_get_query_engine():
        return MagicMock()

    def override_get_mcp_manager():
        return MagicMock()

    # Override database dependency
    app.dependency_overrides[deps.get_db] = override_get_db
    app.dependency_overrides[deps.get_inference_service] = (
        override_get_inference_service
    )
    app.dependency_overrides[deps.get_query_engine] = override_get_query_engine
    app.dependency_overrides[get_mcp_manager] = override_get_mcp_manager

    # Set agent_factory and query_engine in app.state for access in endpoints
    app.state.agent_factory = mock_agent_factory
    app.state.query_engine = override_get_query_engine()

    # For non-dependency-injected components, use patching
    # We patch the modules where these are used
    with (
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_sessions.session_manager",
            mock_session_manager,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_stream.session_manager",
            mock_session_manager,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_stream.get_agent_factory",
            return_value=mock_agent_factory,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_history.get_agent_factory",
            return_value=mock_agent_factory,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.chat.chat_history.get_query_engine",
            return_value=override_get_query_engine(),
        ),
    ):
        with TestClient(app) as client:
            yield client

    # Clean up dependency overrides
    app.dependency_overrides.clear()


@pytest.fixture
def api_client_with_inference(mock_db_session, mock_inference_service):
    """
    FastAPI TestClient with inference service mocked.

    Use this fixture for inference endpoint tests that need:
    - Mock database session
    - Mock inference service (no actual model loading)

    Args:
        mock_db_session: Mock database session fixture
        mock_inference_service: Mock inference service fixture

    Yields:
        TestClient: FastAPI test client with inference mocked
    """

    def override_get_db():
        yield mock_db_session

    def override_get_inference_service():
        return mock_inference_service

    app.dependency_overrides[deps.get_db] = override_get_db
    app.dependency_overrides[deps.get_inference_service] = (
        override_get_inference_service
    )

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_run_data():
    """
    Sample run data for testing run endpoints.

    Returns a dictionary with typical run attributes that can be used
    to populate mock database responses or test payload validation.

    Returns:
        dict: Sample run data with typical fields
    """
    return {
        "id": 1,
        "run_description": "Test Run",
        "status": "completed",
        "training_mode": "centralized",
        "start_time": datetime.datetime(2023, 1, 1, 10, 0, 0),
        "end_time": datetime.datetime(2023, 1, 1, 11, 0, 0),
    }


@pytest.fixture
def sample_metrics_data():
    """
    Sample metrics data for testing analytics endpoints.

    Returns a list of metric dictionaries that can be used to populate
    mock database responses for analytics queries.

    Returns:
        list: List of sample metric dictionaries
    """
    return [
        {"run_id": 1, "metric_name": "val_acc", "metric_value": 0.85, "step": 1},
        {"run_id": 1, "metric_name": "val_loss", "metric_value": 0.30, "step": 1},
        {"run_id": 1, "metric_name": "train_acc", "metric_value": 0.90, "step": 1},
        {"run_id": 1, "metric_name": "train_loss", "metric_value": 0.25, "step": 1},
    ]


@pytest.fixture
def mock_facade():
    """
    Mock AnalyticsFacade for runs endpoints.

    Simulates the AnalyticsFacade behavior for analytics operations:
    - summary: List runs with summaries
    - metrics: Get run metrics
    - backfill: Backfill operations
    - export: Export operations

    Returns:
        MagicMock: Mock analytics facade with predefined behaviors
    """
    mock = MagicMock()
    mock.metrics = MagicMock()
    mock.summary = MagicMock()
    mock.backfill = MagicMock()
    mock.export = MagicMock()

    # Default: return empty results
    mock.summary.list_runs_with_summaries.return_value = {"runs": [], "total": 0}
    mock.metrics.get_run_metrics.return_value = []

    return mock


@pytest.fixture
def mock_run_crud():
    """
    Mock RunCRUD for runs endpoints.

    Simulates RunCRUD operations for database interactions:
    - get_multi: Get multiple runs
    - get: Get single run by ID
    - create: Create new run

    Returns:
        MagicMock: Mock run CRUD with predefined behaviors
    """
    from federated_pneumonia_detection.src.boundary.CRUD.run import RunCRUD

    mock = MagicMock(spec=RunCRUD)
    mock.get_multi.return_value = []
    mock.get.return_value = None
    return mock


@pytest.fixture
def dummy_image_bytes():
    """
    Create a valid 1x1 pixel JPEG image for testing inference endpoints.

    Returns:
        bytes: JPEG image bytes
    """
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (1, 1), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_inference_engine():
    """
    Mock InferenceEngine for inference endpoint tests.

    Simulates InferenceEngine behavior:
    - predict: Returns prediction tuple
    - get_info: Returns engine info
    - model_version: Version string

    Returns:
        MagicMock: Mock inference engine with sensible defaults
    """
    mock = MagicMock()
    mock.model_version = "mock-1.0.0"
    mock.get_info.return_value = {"gpu_available": False, "model_version": "mock-1.0.0"}
    # Default prediction: NORMAL, Confidence, Pneumonia Prob, Normal Prob
    mock.predict.return_value = ("NORMAL", 0.99, 0.01, 0.99)
    return mock


@pytest.fixture
def mock_orchestrator():
    """
    Mock StreamingOrchestrator (Agent) for chat endpoints.

    Simulates agent behavior for chat operations:
    - stream: Async generator yielding chat events
    - history: Returns conversation history
    - query: Returns response

    Returns:
        MagicMock: Mock orchestrator with streaming support
    """
    mock = MagicMock()

    # Mock the stream method which is an async generator
    async def mock_stream(*args, **kwargs):
        yield {"type": "session", "session_id": "test-session-id"}
        yield {"type": "token", "content": "Hello"}
        yield {"type": "token", "content": " world"}
        yield {"type": "done"}

    mock.stream = mock_stream
    mock.history.return_value = [("User message", "AI response")]
    return mock
