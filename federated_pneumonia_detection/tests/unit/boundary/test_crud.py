"""
Comprehensive unit tests for CRUD operations.

Tests all CRUD classes with full coverage:
- BaseCRUD: Base CRUD operations
- ExperimentCRUD: Experiment-specific operations
- RunCRUD: Run-specific operations
- RunConfigurationCRUD: Configuration-specific operations
- RunMetricCRUD: Metric-specific operations
- RunArtifactCRUD: Artifact-specific operations
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.CRUD.experiment import ExperimentCRUD, experiment_crud
from federated_pneumonia_detection.src.boundary.CRUD.run import RunCRUD, run_crud
from federated_pneumonia_detection.src.boundary.CRUD.run_configuration import RunConfigurationCRUD, run_configuration_crud
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import RunMetricCRUD, run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.run_artifact import RunArtifactCRUD, run_artifact_crud
from federated_pneumonia_detection.src.boundary.engine import Experiment, Run, RunConfiguration, RunMetric, RunArtifact


class TestBaseCRUD:
    """Test suite for BaseCRUD class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model class."""
        return Mock()
    
    @pytest.fixture
    def base_crud(self, mock_model):
        """Create a BaseCRUD instance with mock model."""
        return BaseCRUD(mock_model)
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return Mock()
    
    def test_init(self, mock_model):
        """Test BaseCRUD initialization."""
        crud = BaseCRUD(mock_model)
        assert crud.model == mock_model
    
    def test_get_db_session_success(self, base_crud):
        """Test get_db_session context manager on success."""
        mock_session = Mock()
        with patch('federated_pneumonia_detection.src.boundary.CRUD.base.get_session', return_value=mock_session):
            with base_crud.get_db_session() as session:
                assert session == mock_session
            
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()
    
    def test_get_db_session_rollback_on_error(self, base_crud):
        """Test get_db_session rollback on SQLAlchemyError."""
        mock_session = Mock()
        mock_session.commit.side_effect = SQLAlchemyError("DB Error")
        
        with patch('federated_pneumonia_detection.src.boundary.CRUD.base.get_session', return_value=mock_session):
            with pytest.raises(SQLAlchemyError):
                with base_crud.get_db_session() as session:
                    pass
            
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()
    
    def test_create(self, base_crud, mock_db):
        """Test create operation."""
        mock_obj = Mock()
        base_crud.model.return_value = mock_obj
        
        result = base_crud.create(mock_db, name="test", value=123)
        
        base_crud.model.assert_called_once_with(name="test", value=123)
        mock_db.add.assert_called_once_with(mock_obj)
        mock_db.flush.assert_called_once()
        mock_db.refresh.assert_called_once_with(mock_obj)
        assert result == mock_obj
    
    def test_get_found(self, base_crud, mock_db):
        """Test get operation when record exists."""
        mock_query = Mock()
        mock_filter = Mock()
        mock_obj = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_obj
        
        result = base_crud.get(mock_db, 1)
        
        mock_db.query.assert_called_once_with(base_crud.model)
        assert result == mock_obj
    
    def test_get_not_found(self, base_crud, mock_db):
        """Test get operation when record doesn't exist."""
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        
        result = base_crud.get(mock_db, 999)
        
        assert result is None
    
    def test_get_multi_without_filters(self, base_crud, mock_db):
        """Test get_multi without filters."""
        mock_query = Mock()
        mock_offset = Mock()
        mock_limit = Mock()
        mock_objs = [Mock(), Mock()]
        
        mock_db.query.return_value = mock_query
        mock_query.offset.return_value = mock_offset
        mock_offset.limit.return_value = mock_limit
        mock_limit.all.return_value = mock_objs
        
        result = base_crud.get_multi(mock_db, skip=0, limit=100)
        
        mock_query.offset.assert_called_once_with(0)
        mock_offset.limit.assert_called_once_with(100)
        assert result == mock_objs
    
    def test_get_multi_with_filters(self, base_crud, mock_db):
        """Test get_multi with filters."""
        mock_query = Mock()
        mock_filtered = Mock()
        mock_offset = Mock()
        mock_limit = Mock()
        mock_objs = [Mock()]
        
        mock_db.query.return_value = mock_query
        base_crud.model.status = Mock()
        mock_query.filter.return_value = mock_filtered
        mock_filtered.offset.return_value = mock_offset
        mock_offset.limit.return_value = mock_limit
        mock_limit.all.return_value = mock_objs
        
        filters = {'status': 'active'}
        result = base_crud.get_multi(mock_db, filters=filters)
        
        mock_query.filter.assert_called_once()
        assert result == mock_objs
    
    def test_get_multi_with_invalid_filter_key(self, base_crud, mock_db):
        """Test get_multi skips invalid filter keys."""
        mock_query = Mock()
        mock_offset = Mock()
        mock_limit = Mock()
        mock_objs = [Mock()]
        
        mock_db.query.return_value = mock_query
        mock_query.offset.return_value = mock_offset
        mock_offset.limit.return_value = mock_limit
        mock_limit.all.return_value = mock_objs
        
        base_crud.model.invalid_key = None
        base_crud.model.__dict__ = {}
        
        filters = {'invalid_key': 'value'}
        result = base_crud.get_multi(mock_db, filters=filters)
        
        assert result == mock_objs
    
    def test_update_found(self, base_crud, mock_db):
        """Test update operation when record exists."""
        mock_obj = Mock()
        mock_obj.name = "old_name"
        base_crud.get = Mock(return_value=mock_obj)
        
        result = base_crud.update(mock_db, 1, name="new_name")
        
        assert mock_obj.name == "new_name"
        mock_db.flush.assert_called_once()
        mock_db.refresh.assert_called_once_with(mock_obj)
        assert result == mock_obj
    
    def test_update_not_found(self, base_crud, mock_db):
        """Test update operation when record doesn't exist."""
        base_crud.get = Mock(return_value=None)
        
        result = base_crud.update(mock_db, 999, name="new_name")
        
        assert result is None
    
    def test_update_ignores_invalid_attributes(self, base_crud, mock_db):
        """Test update ignores attributes that don't exist."""
        mock_obj = Mock(spec=['name'])
        mock_obj.name = "old_name"
        base_crud.get = Mock(return_value=mock_obj)
        
        result = base_crud.update(mock_db, 1, name="new_name", invalid_attr="value")
        
        assert mock_obj.name == "new_name"
        assert result == mock_obj
    
    def test_delete_found(self, base_crud, mock_db):
        """Test delete operation when record exists."""
        mock_obj = Mock()
        base_crud.get = Mock(return_value=mock_obj)
        
        result = base_crud.delete(mock_db, 1)
        
        mock_db.delete.assert_called_once_with(mock_obj)
        mock_db.flush.assert_called_once()
        assert result is True
    
    def test_delete_not_found(self, base_crud, mock_db):
        """Test delete operation when record doesn't exist."""
        base_crud.get = Mock(return_value=None)
        
        result = base_crud.delete(mock_db, 999)
        
        mock_db.delete.assert_not_called()
        assert result is False
    
    def test_count_without_filters(self, base_crud, mock_db):
        """Test count without filters."""
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.count.return_value = 42
        
        result = base_crud.count(mock_db)
        
        mock_db.query.assert_called_once_with(base_crud.model)
        mock_query.count.assert_called_once()
        assert result == 42
    
    def test_count_with_filters(self, base_crud, mock_db):
        """Test count with filters."""
        mock_query = Mock()
        mock_filtered = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filtered
        mock_filtered.count.return_value = 10
        
        filters = {'status': 'active'}
        result = base_crud.count(mock_db, filters=filters)
        
        mock_query.filter.assert_called_once()
        assert result == 10
    
    def test_exists_true(self, base_crud, mock_db):
        """Test exists when record is found."""
        mock_query = Mock()
        mock_filter = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = Mock()
        
        result = base_crud.exists(mock_db, 1)
        
        assert result is True
    
    def test_exists_false(self, base_crud, mock_db):
        """Test exists when record is not found."""
        mock_query = Mock()
        mock_filter = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        
        result = base_crud.exists(mock_db, 999)
        
        assert result is False
    
    def test_bulk_create(self, base_crud, mock_db):
        """Test bulk_create operation."""
        mock_objs = [Mock(), Mock(), Mock()]
        base_crud.model.side_effect = mock_objs
        
        objects_data = [{'name': 'obj1'}, {'name': 'obj2'}, {'name': 'obj3'}]
        result = base_crud.bulk_create(mock_db, objects_data)
        
        mock_db.add_all.assert_called_once()
        mock_db.flush.assert_called_once()
        assert len(result) == 3
    
    def test_bulk_create_empty_list(self, base_crud, mock_db):
        """Test bulk_create with empty list."""
        result = base_crud.bulk_create(mock_db, [])
        
        mock_db.add_all.assert_called_once_with([])
        assert result == []


class TestExperimentCRUD:
    """Test suite for ExperimentCRUD class."""
    
    @pytest.fixture
    def exp_crud(self):
        """Create ExperimentCRUD instance."""
        return ExperimentCRUD()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()
    
    def test_init(self):
        """Test ExperimentCRUD initialization."""
        crud = ExperimentCRUD()
        assert crud.model == Experiment
    
    def test_singleton_instance(self):
        """Test that experiment_crud singleton exists."""
        assert isinstance(experiment_crud, ExperimentCRUD)
        assert experiment_crud.model == Experiment
    
    def test_get_by_name(self, exp_crud, mock_db):
        """Test get_by_name method."""
        mock_exp = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_exp
        
        result = exp_crud.get_by_name(mock_db, "test_exp")
        
        mock_db.query.assert_called_once_with(Experiment)
        assert result == mock_exp
    
    def test_get_by_name_not_found(self, exp_crud, mock_db):
        """Test get_by_name when experiment doesn't exist."""
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        
        result = exp_crud.get_by_name(mock_db, "nonexistent")
        
        assert result is None
    
    def test_get_with_runs(self, exp_crud, mock_db):
        """Test get_with_runs method."""
        mock_exp = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_exp
        
        result = exp_crud.get_with_runs(mock_db, 1)
        
        assert result == mock_exp
    
    def test_get_recent(self, exp_crud, mock_db):
        """Test get_recent method."""
        mock_exps = [Mock(), Mock()]
        mock_query = Mock()
        mock_ordered = Mock()
        mock_limited = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.order_by.return_value = mock_ordered
        mock_ordered.limit.return_value = mock_limited
        mock_limited.all.return_value = mock_exps
        
        result = exp_crud.get_recent(mock_db, limit=10)
        
        mock_query.order_by.assert_called_once()
        mock_ordered.limit.assert_called_once_with(10)
        assert result == mock_exps
    
    def test_get_recent_default_limit(self, exp_crud, mock_db):
        """Test get_recent with default limit."""
        mock_exps = [Mock()]
        mock_query = Mock()
        mock_ordered = Mock()
        mock_limited = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.order_by.return_value = mock_ordered
        mock_ordered.limit.return_value = mock_limited
        mock_limited.all.return_value = mock_exps
        
        result = exp_crud.get_recent(mock_db)
        
        mock_ordered.limit.assert_called_once_with(10)
        assert result == mock_exps
    
    def test_search_by_name(self, exp_crud, mock_db):
        """Test search_by_name method."""
        mock_exps = [Mock(), Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_exps
        
        result = exp_crud.search_by_name(mock_db, "test")
        
        mock_query.filter.assert_called_once()
        assert result == mock_exps
    
    def test_search_by_name_empty_result(self, exp_crud, mock_db):
        """Test search_by_name with no matches."""
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = []
        
        result = exp_crud.search_by_name(mock_db, "nonexistent_pattern")
        
        assert result == []


class TestRunCRUD:
    """Test suite for RunCRUD class."""
    
    @pytest.fixture
    def run_crud_instance(self):
        """Create RunCRUD instance."""
        return RunCRUD()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()
    
    def test_init(self):
        """Test RunCRUD initialization."""
        crud = RunCRUD()
        assert crud.model == Run
    
    def test_singleton_instance(self):
        """Test that run_crud singleton exists."""
        assert isinstance(run_crud, RunCRUD)
        assert run_crud.model == Run
    
    def test_get_by_experiment(self, run_crud_instance, mock_db):
        """Test get_by_experiment method."""
        mock_runs = [Mock(), Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_runs
        
        result = run_crud_instance.get_by_experiment(mock_db, 1)
        
        mock_query.filter.assert_called_once()
        assert result == mock_runs
    
    def test_get_by_status(self, run_crud_instance, mock_db):
        """Test get_by_status method."""
        mock_runs = [Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_runs
        
        result = run_crud_instance.get_by_status(mock_db, "completed")
        
        assert result == mock_runs
    
    def test_get_by_training_mode(self, run_crud_instance, mock_db):
        """Test get_by_training_mode method."""
        mock_runs = [Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_runs
        
        result = run_crud_instance.get_by_training_mode(mock_db, "federated")
        
        assert result == mock_runs
    
    def test_get_with_config(self, run_crud_instance, mock_db):
        """Test get_with_config method."""
        mock_run = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_run
        
        result = run_crud_instance.get_with_config(mock_db, 1)
        
        assert result == mock_run
    
    def test_get_with_metrics(self, run_crud_instance, mock_db):
        """Test get_with_metrics method."""
        mock_run = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_run
        
        result = run_crud_instance.get_with_metrics(mock_db, 1)
        
        assert result == mock_run
    
    def test_get_with_artifacts(self, run_crud_instance, mock_db):
        """Test get_with_artifacts method."""
        mock_run = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_run
        
        result = run_crud_instance.get_with_artifacts(mock_db, 1)
        
        assert result == mock_run
    
    def test_update_status(self, run_crud_instance, mock_db):
        """Test update_status method."""
        mock_run = Mock()
        mock_run.status = "pending"
        run_crud_instance.update = Mock(return_value=mock_run)
        
        result = run_crud_instance.update_status(mock_db, 1, "completed")
        
        run_crud_instance.update.assert_called_once_with(mock_db, 1, status="completed")
        assert result == mock_run
    
    def test_get_by_wandb_id(self, run_crud_instance, mock_db):
        """Test get_by_wandb_id method."""
        mock_run = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_run
        
        result = run_crud_instance.get_by_wandb_id(mock_db, "wandb_id_123")
        
        assert result == mock_run
    
    def test_get_by_wandb_id_not_found(self, run_crud_instance, mock_db):
        """Test get_by_wandb_id when not found."""
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        
        result = run_crud_instance.get_by_wandb_id(mock_db, "nonexistent")
        
        assert result is None
    
    def test_get_completed_runs_all(self, run_crud_instance, mock_db):
        """Test get_completed_runs for all experiments."""
        mock_runs = [Mock(), Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        mock_ordered = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_ordered
        mock_ordered.all.return_value = mock_runs
        
        result = run_crud_instance.get_completed_runs(mock_db)
        
        assert result == mock_runs
    
    def test_get_completed_runs_by_experiment(self, run_crud_instance, mock_db):
        """Test get_completed_runs filtered by experiment."""
        mock_runs = [Mock()]
        mock_query = Mock()
        mock_filter1 = Mock()
        mock_filter2 = Mock()
        mock_ordered = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter1
        mock_filter1.filter.return_value = mock_filter2
        mock_filter2.order_by.return_value = mock_ordered
        mock_ordered.all.return_value = mock_runs
        
        result = run_crud_instance.get_completed_runs(mock_db, experiment_id=1)
        
        assert result == mock_runs
    
    def test_get_failed_runs_all(self, run_crud_instance, mock_db):
        """Test get_failed_runs for all experiments."""
        mock_runs = [Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        mock_ordered = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_ordered
        mock_ordered.all.return_value = mock_runs
        
        result = run_crud_instance.get_failed_runs(mock_db)
        
        assert result == mock_runs
    
    def test_get_failed_runs_by_experiment(self, run_crud_instance, mock_db):
        """Test get_failed_runs filtered by experiment."""
        mock_runs = [Mock()]
        mock_query = Mock()
        mock_filter1 = Mock()
        mock_filter2 = Mock()
        mock_ordered = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter1
        mock_filter1.filter.return_value = mock_filter2
        mock_filter2.order_by.return_value = mock_ordered
        mock_ordered.all.return_value = mock_runs
        
        result = run_crud_instance.get_failed_runs(mock_db, experiment_id=1)
        
        assert result == mock_runs


class TestRunConfigurationCRUD:
    """Test suite for RunConfigurationCRUD class."""
    
    @pytest.fixture
    def config_crud(self):
        """Create RunConfigurationCRUD instance."""
        return RunConfigurationCRUD()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()
    
    def test_init(self):
        """Test RunConfigurationCRUD initialization."""
        crud = RunConfigurationCRUD()
        assert crud.model == RunConfiguration
    
    def test_singleton_instance(self):
        """Test that run_configuration_crud singleton exists."""
        assert isinstance(run_configuration_crud, RunConfigurationCRUD)
        assert run_configuration_crud.model == RunConfiguration
    
    def test_get_by_run(self, config_crud, mock_db):
        """Test get_by_run method."""
        mock_config = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_config
        
        result = config_crud.get_by_run(mock_db, 1)
        
        mock_query.filter.assert_called_once()
        assert result == mock_config
    
    def test_get_by_run_not_found(self, config_crud, mock_db):
        """Test get_by_run when not found."""
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        
        result = config_crud.get_by_run(mock_db, 999)
        
        assert result is None
    
    def test_get_by_partition_strategy(self, config_crud, mock_db):
        """Test get_by_partition_strategy method."""
        mock_configs = [Mock(), Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_configs
        
        result = config_crud.get_by_partition_strategy(mock_db, "iid")
        
        assert result == mock_configs
    
    def test_get_by_learning_rate(self, config_crud, mock_db):
        """Test get_by_learning_rate method."""
        mock_configs = [Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_configs
        
        result = config_crud.get_by_learning_rate(mock_db, 0.001)
        
        assert result == mock_configs
    
    def test_get_by_hyperparameters_all(self, config_crud, mock_db):
        """Test get_by_hyperparameters with all parameters."""
        mock_configs = [Mock()]
        mock_query = Mock()
        mock_filter1 = Mock()
        mock_filter2 = Mock()
        mock_filter3 = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter1
        mock_filter1.filter.return_value = mock_filter2
        mock_filter2.filter.return_value = mock_filter3
        mock_filter3.all.return_value = mock_configs
        
        result = config_crud.get_by_hyperparameters(
            mock_db, 
            learning_rate=0.001, 
            batch_size=32, 
            epochs=100
        )
        
        assert result == mock_configs
    
    def test_get_by_hyperparameters_partial(self, config_crud, mock_db):
        """Test get_by_hyperparameters with partial parameters."""
        mock_configs = [Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_configs
        
        result = config_crud.get_by_hyperparameters(mock_db, learning_rate=0.001)
        
        assert result == mock_configs
    
    def test_get_by_hyperparameters_none(self, config_crud, mock_db):
        """Test get_by_hyperparameters with no parameters."""
        mock_configs = [Mock()]
        mock_query = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.all.return_value = mock_configs
        
        result = config_crud.get_by_hyperparameters(mock_db)
        
        assert result == mock_configs


class TestRunMetricCRUD:
    """Test suite for RunMetricCRUD class."""
    
    @pytest.fixture
    def metric_crud(self):
        """Create RunMetricCRUD instance."""
        return RunMetricCRUD()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()
    
    def test_init(self):
        """Test RunMetricCRUD initialization."""
        crud = RunMetricCRUD()
        assert crud.model == RunMetric
    
    def test_singleton_instance(self):
        """Test that run_metric_crud singleton exists."""
        assert isinstance(run_metric_crud, RunMetricCRUD)
        assert run_metric_crud.model == RunMetric
    
    def test_get_by_run(self, metric_crud, mock_db):
        """Test get_by_run method."""
        mock_metrics = [Mock(), Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_metrics
        
        result = metric_crud.get_by_run(mock_db, 1)
        
        assert result == mock_metrics
    
    def test_get_by_run_empty(self, metric_crud, mock_db):
        """Test get_by_run with no metrics."""
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = []
        
        result = metric_crud.get_by_run(mock_db, 999)
        
        assert result == []
    
    def test_get_by_metric_name(self, metric_crud, mock_db):
        """Test get_by_metric_name method."""
        mock_metrics = [Mock(), Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        mock_ordered = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_ordered
        mock_ordered.all.return_value = mock_metrics
        
        result = metric_crud.get_by_metric_name(mock_db, 1, "accuracy")
        
        mock_filter.order_by.assert_called_once()
        assert result == mock_metrics
    
    def test_get_by_dataset_type(self, metric_crud, mock_db):
        """Test get_by_dataset_type method."""
        mock_metrics = [Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_metrics
        
        result = metric_crud.get_by_dataset_type(mock_db, 1, "test")
        
        assert result == mock_metrics
    
    def test_get_latest_by_metric(self, metric_crud, mock_db):
        """Test get_latest_by_metric method."""
        mock_metric = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        mock_ordered = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_ordered
        mock_ordered.first.return_value = mock_metric
        
        result = metric_crud.get_latest_by_metric(mock_db, 1, "accuracy")
        
        assert result == mock_metric
    
    def test_get_by_step(self, metric_crud, mock_db):
        """Test get_by_step method."""
        mock_metrics = [Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_metrics
        
        result = metric_crud.get_by_step(mock_db, 1, 100)
        
        assert result == mock_metrics
    
    def test_get_best_metric_maximize(self, metric_crud, mock_db):
        """Test get_best_metric with maximize=True."""
        mock_metric = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        mock_ordered = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_ordered
        mock_ordered.first.return_value = mock_metric
        
        result = metric_crud.get_best_metric(mock_db, 1, "accuracy", maximize=True)
        
        assert result == mock_metric
    
    def test_get_best_metric_minimize(self, metric_crud, mock_db):
        """Test get_best_metric with maximize=False."""
        mock_metric = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        mock_ordered = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_ordered
        mock_ordered.first.return_value = mock_metric
        
        result = metric_crud.get_best_metric(mock_db, 1, "loss", maximize=False)
        
        assert result == mock_metric
    
    def test_get_metric_stats_empty(self, metric_crud, mock_db):
        """Test get_metric_stats with no metrics."""
        metric_crud.get_by_metric_name = Mock(return_value=[])
        
        result = metric_crud.get_metric_stats(mock_db, 1, "accuracy")
        
        assert result == {}
    
    def test_get_metric_stats_with_data(self, metric_crud, mock_db):
        """Test get_metric_stats with metrics."""
        mock_m1 = Mock(metric_value=0.8)
        mock_m2 = Mock(metric_value=0.85)
        mock_m3 = Mock(metric_value=0.9)
        metric_crud.get_by_metric_name = Mock(return_value=[mock_m1, mock_m2, mock_m3])
        
        result = metric_crud.get_metric_stats(mock_db, 1, "accuracy")
        
        assert result["min"] == 0.8
        assert result["max"] == 0.9
        assert result["mean"] == pytest.approx(0.85)
        assert result["count"] == 3
        assert result["latest"] == 0.9
    
    def test_get_metrics_by_name_and_dataset(self, metric_crud, mock_db):
        """Test get_metrics_by_name_and_dataset method."""
        mock_metrics = [Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        mock_ordered = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_ordered
        mock_ordered.all.return_value = mock_metrics
        
        result = metric_crud.get_metrics_by_name_and_dataset(
            mock_db, 1, "accuracy", "test"
        )
        
        assert result == mock_metrics


class TestRunArtifactCRUD:
    """Test suite for RunArtifactCRUD class."""
    
    @pytest.fixture
    def artifact_crud(self):
        """Create RunArtifactCRUD instance."""
        return RunArtifactCRUD()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()
    
    def test_init(self):
        """Test RunArtifactCRUD initialization."""
        crud = RunArtifactCRUD()
        assert crud.model == RunArtifact
    
    def test_singleton_instance(self):
        """Test that run_artifact_crud singleton exists."""
        assert isinstance(run_artifact_crud, RunArtifactCRUD)
        assert run_artifact_crud.model == RunArtifact
    
    def test_get_by_run(self, artifact_crud, mock_db):
        """Test get_by_run method."""
        mock_artifacts = [Mock(), Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_artifacts
        
        result = artifact_crud.get_by_run(mock_db, 1)
        
        assert result == mock_artifacts
    
    def test_get_by_run_empty(self, artifact_crud, mock_db):
        """Test get_by_run with no artifacts."""
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = []
        
        result = artifact_crud.get_by_run(mock_db, 999)
        
        assert result == []
    
    def test_get_by_type(self, artifact_crud, mock_db):
        """Test get_by_type method."""
        mock_artifacts = [Mock(), Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_artifacts
        
        result = artifact_crud.get_by_type(mock_db, 1, "model")
        
        assert result == mock_artifacts
    
    def test_get_by_name(self, artifact_crud, mock_db):
        """Test get_by_name method."""
        mock_artifact = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_artifact
        
        result = artifact_crud.get_by_name(mock_db, 1, "model.pt")
        
        assert result == mock_artifact
    
    def test_get_by_name_not_found(self, artifact_crud, mock_db):
        """Test get_by_name when not found."""
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        
        result = artifact_crud.get_by_name(mock_db, 1, "nonexistent.pt")
        
        assert result is None
    
    def test_get_models(self, artifact_crud, mock_db):
        """Test get_models method."""
        mock_artifacts = [Mock()]
        artifact_crud.get_by_type = Mock(return_value=mock_artifacts)
        
        result = artifact_crud.get_models(mock_db, 1)
        
        artifact_crud.get_by_type.assert_called_once_with(mock_db, 1, 'model')
        assert result == mock_artifacts
    
    def test_get_images(self, artifact_crud, mock_db):
        """Test get_images method."""
        mock_artifacts = [Mock()]
        artifact_crud.get_by_type = Mock(return_value=mock_artifacts)
        
        result = artifact_crud.get_images(mock_db, 1)
        
        artifact_crud.get_by_type.assert_called_once_with(mock_db, 1, 'image')
        assert result == mock_artifacts
    
    def test_get_logs(self, artifact_crud, mock_db):
        """Test get_logs method."""
        mock_artifacts = [Mock()]
        artifact_crud.get_by_type = Mock(return_value=mock_artifacts)
        
        result = artifact_crud.get_logs(mock_db, 1)
        
        artifact_crud.get_by_type.assert_called_once_with(mock_db, 1, 'log')
        assert result == mock_artifacts
    
    def test_search_by_name(self, artifact_crud, mock_db):
        """Test search_by_name method."""
        mock_artifacts = [Mock()]
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = mock_artifacts
        
        result = artifact_crud.search_by_name(mock_db, 1, "model")
        
        mock_query.filter.assert_called_once()
        assert result == mock_artifacts
    
    def test_search_by_name_empty(self, artifact_crud, mock_db):
        """Test search_by_name with no matches."""
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.all.return_value = []
        
        result = artifact_crud.search_by_name(mock_db, 1, "nonexistent")
        
        assert result == []


class TestCRUDIntegration:
    """Integration tests for CRUD classes."""
    
    def test_experiment_crud_inheritance(self):
        """Test ExperimentCRUD inherits from BaseCRUD."""
        crud = ExperimentCRUD()
        assert isinstance(crud, BaseCRUD)
        assert hasattr(crud, 'create')
        assert hasattr(crud, 'get')
        assert hasattr(crud, 'get_multi')
        assert hasattr(crud, 'update')
        assert hasattr(crud, 'delete')
    
    def test_run_crud_inheritance(self):
        """Test RunCRUD inherits from BaseCRUD."""
        crud = RunCRUD()
        assert isinstance(crud, BaseCRUD)
        assert hasattr(crud, 'create')
        assert hasattr(crud, 'get')
        assert hasattr(crud, 'get_multi')
        assert hasattr(crud, 'update')
        assert hasattr(crud, 'delete')
    
    def test_all_crud_classes_defined(self):
        """Test all CRUD classes are properly instantiated."""
        assert isinstance(experiment_crud, ExperimentCRUD)
        assert isinstance(run_crud, RunCRUD)
        assert isinstance(run_configuration_crud, RunConfigurationCRUD)
        assert isinstance(run_metric_crud, RunMetricCRUD)
        assert isinstance(run_artifact_crud, RunArtifactCRUD)
    
    def test_all_crud_models_set(self):
        """Test all CRUD instances have correct models."""
        assert experiment_crud.model == Experiment
        assert run_crud.model == Run
        assert run_configuration_crud.model == RunConfiguration
        assert run_metric_crud.model == RunMetric
        assert run_artifact_crud.model == RunArtifact
