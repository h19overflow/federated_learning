import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session."""
    return MagicMock()


@pytest.fixture
def mock_cache():
    """Mock CacheProvider."""
    cache = MagicMock()
    # Mock get_or_set to just call the compute function
    cache.get_or_set.side_effect = lambda key, func: func()
    return cache


@pytest.fixture
def mock_run_crud():
    """Mock RunCRUD."""
    return MagicMock()


@pytest.fixture
def mock_run_metric_crud():
    """Mock RunMetricCRUD."""
    return MagicMock()


@pytest.fixture
def mock_server_eval_crud():
    """Mock ServerEvaluationCRUD."""
    return MagicMock()


@pytest.fixture
def sample_run():
    """Static sample run fixture."""
    now = datetime(2023, 1, 1, 12, 0, 0)
    run = MagicMock()
    run.id = 1
    run.training_mode = "centralized"
    run.status = "completed"
    run.start_time = now
    run.end_time = now + timedelta(minutes=30)
    return run


@pytest.fixture
def sample_federated_run():
    """Static sample federated run fixture."""
    now = datetime(2023, 1, 1, 12, 0, 0)
    run = MagicMock()
    run.id = 2
    run.training_mode = "federated"
    run.status = "completed"
    run.start_time = now
    run.end_time = now + timedelta(minutes=45)
    return run
