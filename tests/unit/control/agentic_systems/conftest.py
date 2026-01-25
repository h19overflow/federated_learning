import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_db_session():
    """MagicMock for SQLAlchemy session."""
    return MagicMock()


@pytest.fixture
def mock_llm():
    """MagicMock for LangChain LLM."""
    return MagicMock()


@pytest.fixture
def mock_vector_store():
    """MagicMock for PGVector."""
    return MagicMock()
