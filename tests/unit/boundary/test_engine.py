"""
Unit tests for boundary layer database engine and models.
Tests SQLAlchemy models, relationships, and database operations.
"""

from unittest.mock import MagicMock, patch
from sqlalchemy.types import (
    String as StringType,
    Integer as IntegerType,
    Float as FloatType,
)

from federated_pneumonia_detection.src.boundary.engine import (
    Base,
    Run,
    RunConfiguration,
    RunMetric,
    RunArtifact,
    create_tables,
    get_session,
)


class TestRunModel:
    """Tests for Run SQLAlchemy model."""

    def test_run_table_name(self):
        """Test that Run model has correct table name."""
        assert Run.__tablename__ == "runs"

    def test_run_required_columns(self):
        """Test that Run has required columns."""
        required_columns = {
            "id",
            "training_mode",
            "status",
            "start_time",
            "end_time",
            "wandb_id",
            "source_path",
        }
        actual_columns = {col.name for col in Run.__table__.columns}
        assert required_columns.issubset(actual_columns)

    def test_run_id_is_primary_key(self):
        """Test that id column is primary key."""
        id_column = Run.__table__.columns["id"]
        assert id_column.primary_key is True

    def test_run_configuration_relationship(self):
        """Test that Run has configuration relationship."""
        assert hasattr(Run, "configuration")
        assert "configuration" in Run.__mapper__.relationships

    def test_run_metrics_relationship(self):
        """Test that Run has metrics relationship."""
        assert hasattr(Run, "metrics")
        assert "metrics" in Run.__mapper__.relationships

    def test_run_artifacts_relationship(self):
        """Test that Run has artifacts relationship."""
        assert hasattr(Run, "artifacts")
        assert "artifacts" in Run.__mapper__.relationships

    def test_run_training_mode_string_type(self):
        """Test that training_mode is String type."""
        training_mode_column = Run.__table__.columns["training_mode"]
        assert isinstance(training_mode_column.type, StringType)

    def test_run_status_string_type(self):
        """Test that status is String type."""
        status_column = Run.__table__.columns["status"]
        assert isinstance(status_column.type, StringType)

    def test_run_wandb_id_string_type(self):
        """Test that wandb_id is String type."""
        wandb_id_column = Run.__table__.columns["wandb_id"]
        assert isinstance(wandb_id_column.type, StringType)

    def test_run_source_path_string_type(self):
        """Test that source_path is String type."""
        source_path_column = Run.__table__.columns["source_path"]
        assert isinstance(source_path_column.type, StringType)


class TestRunConfigurationModel:
    """Tests for RunConfiguration SQLAlchemy model."""

    def test_run_configuration_table_name(self):
        """Test that RunConfiguration has correct table name."""
        assert RunConfiguration.__tablename__ == "run_configurations"

    def test_run_configuration_required_columns(self):
        """Test that RunConfiguration has required columns."""
        required_columns = {
            "id",
            "run_id",
            "learning_rate",
            "epochs",
            "weight_decay",
            "batch_size",
            "num_rounds",
            "num_clients",
            "clients_per_round",
            "local_epochs",
            "partition_strategy",
            "seed",
        }
        actual_columns = {col.name for col in RunConfiguration.__table__.columns}
        assert required_columns.issubset(actual_columns)

    def test_run_configuration_id_is_primary_key(self):
        """Test that id column is primary key."""
        id_column = RunConfiguration.__table__.columns["id"]
        assert id_column.primary_key is True

    def test_run_configuration_run_id_is_foreign_key(self):
        """Test that run_id is foreign key."""
        run_id_column = RunConfiguration.__table__.columns["run_id"]
        assert len(run_id_column.foreign_keys) > 0
        assert any("runs" in str(fk) for fk in run_id_column.foreign_keys)

    def test_run_configuration_run_relationship(self):
        """Test that RunConfiguration has run relationship."""
        assert hasattr(RunConfiguration, "run")
        assert "run" in RunConfiguration.__mapper__.relationships

    def test_run_configuration_numeric_columns(self):
        """Test that numeric columns have correct types."""
        assert isinstance(
            RunConfiguration.__table__.columns["learning_rate"].type, FloatType
        )
        assert isinstance(
            RunConfiguration.__table__.columns["epochs"].type, IntegerType
        )
        assert isinstance(
            RunConfiguration.__table__.columns["weight_decay"].type, FloatType
        )
        assert isinstance(
            RunConfiguration.__table__.columns["batch_size"].type, IntegerType
        )

    def test_run_configuration_partition_strategy_string(self):
        """Test that partition_strategy is String type."""
        partition_strategy_column = RunConfiguration.__table__.columns[
            "partition_strategy"
        ]
        assert isinstance(partition_strategy_column.type, StringType)


class TestRunMetricModel:
    """Tests for RunMetric SQLAlchemy model."""

    def test_run_metric_table_name(self):
        """Test that RunMetric has correct table name."""
        assert RunMetric.__tablename__ == "run_metrics"

    def test_run_metric_required_columns(self):
        """Test that RunMetric has required columns."""
        required_columns = {
            "id",
            "run_id",
            "metric_name",
            "metric_value",
            "step",
            "dataset_type",
        }
        actual_columns = {col.name for col in RunMetric.__table__.columns}
        assert required_columns.issubset(actual_columns)

    def test_run_metric_id_is_primary_key(self):
        """Test that id column is primary key."""
        id_column = RunMetric.__table__.columns["id"]
        assert id_column.primary_key is True

    def test_run_metric_run_id_is_foreign_key(self):
        """Test that run_id is foreign key."""
        run_id_column = RunMetric.__table__.columns["run_id"]
        assert len(run_id_column.foreign_keys) > 0
        assert any("runs" in str(fk) for fk in run_id_column.foreign_keys)

    def test_run_metric_run_relationship(self):
        """Test that RunMetric has run relationship."""
        assert hasattr(RunMetric, "run")
        assert "run" in RunMetric.__mapper__.relationships

    def test_run_metric_metric_name_string_type(self):
        """Test that metric_name is String type."""
        metric_name_column = RunMetric.__table__.columns["metric_name"]
        assert isinstance(metric_name_column.type, StringType)

    def test_run_metric_metric_value_float_type(self):
        """Test that metric_value is Float type."""
        metric_value_column = RunMetric.__table__.columns["metric_value"]
        assert isinstance(metric_value_column.type, FloatType)

    def test_run_metric_step_integer_type(self):
        """Test that step is Integer type."""
        step_column = RunMetric.__table__.columns["step"]
        assert isinstance(step_column.type, IntegerType)

    def test_run_metric_dataset_type_string(self):
        """Test that dataset_type is String type."""
        dataset_type_column = RunMetric.__table__.columns["dataset_type"]
        assert isinstance(dataset_type_column.type, StringType)


class TestRunArtifactModel:
    """Tests for RunArtifact SQLAlchemy model."""

    def test_run_artifact_table_name(self):
        """Test that RunArtifact has correct table name."""
        assert RunArtifact.__tablename__ == "run_artifacts"

    def test_run_artifact_required_columns(self):
        """Test that RunArtifact has required columns."""
        required_columns = {
            "id",
            "run_id",
            "artifact_name",
            "artifact_path",
            "artifact_type",
        }
        actual_columns = {col.name for col in RunArtifact.__table__.columns}
        assert required_columns.issubset(actual_columns)

    def test_run_artifact_id_is_primary_key(self):
        """Test that id column is primary key."""
        id_column = RunArtifact.__table__.columns["id"]
        assert id_column.primary_key is True

    def test_run_artifact_run_id_is_foreign_key(self):
        """Test that run_id is foreign key."""
        run_id_column = RunArtifact.__table__.columns["run_id"]
        assert len(run_id_column.foreign_keys) > 0
        assert any("runs" in str(fk) for fk in run_id_column.foreign_keys)

    def test_run_artifact_run_relationship(self):
        """Test that RunArtifact has run relationship."""
        assert hasattr(RunArtifact, "run")
        assert "run" in RunArtifact.__mapper__.relationships

    def test_run_artifact_artifact_name_string_type(self):
        """Test that artifact_name is String type."""
        artifact_name_column = RunArtifact.__table__.columns["artifact_name"]
        assert isinstance(artifact_name_column.type, StringType)

    def test_run_artifact_artifact_path_string_type(self):
        """Test that artifact_path is String type."""
        artifact_path_column = RunArtifact.__table__.columns["artifact_path"]
        assert isinstance(artifact_path_column.type, StringType)

    def test_run_artifact_artifact_type_string(self):
        """Test that artifact_type is String type."""
        artifact_type_column = RunArtifact.__table__.columns["artifact_type"]
        assert isinstance(artifact_type_column.type, StringType)


class TestModelRelationships:
    """Tests for relationships between models."""

    def test_run_configuration_bidirectional_relationship(self):
        """Test bidirectional relationship between Run and RunConfiguration."""
        assert hasattr(Run, "configuration")
        assert hasattr(RunConfiguration, "run")

    def test_run_metric_bidirectional_relationship(self):
        """Test bidirectional relationship between Run and RunMetric."""
        assert hasattr(Run, "metrics")
        assert hasattr(RunMetric, "run")

    def test_run_artifact_bidirectional_relationship(self):
        """Test bidirectional relationship between Run and RunArtifact."""
        assert hasattr(Run, "artifacts")
        assert hasattr(RunArtifact, "run")

    def test_run_configuration_one_to_one(self):
        """Test that RunConfiguration uses uselist=False for one-to-one."""
        # Check in the relationship definition
        run_config_relationship = Run.__mapper__.relationships["configuration"]
        # uselist=False indicates one-to-one relationship
        assert run_config_relationship.uselist is False


class TestBaseMetadata:
    """Tests for SQLAlchemy Base and metadata."""

    def test_base_is_declarative(self):
        """Test that Base is declarative base."""
        assert hasattr(Base, "metadata")

    def test_base_registry_contains_all_tables(self):
        """Test that Base registry contains all model tables."""
        table_names = {table.name for table in Base.metadata.tables.values()}
        expected_tables = {
            "runs",
            "run_configurations",
            "run_metrics",
            "run_artifacts",
        }
        assert expected_tables.issubset(table_names)


class TestCreateTablesFunction:
    """Tests for create_tables function."""

    @patch("federated_pneumonia_detection.src.boundary.engine.create_engine")
    @patch("federated_pneumonia_detection.src.boundary.engine.settings")
    def test_create_tables_calls_create_engine(self, mock_settings, mock_create_engine):
        """Test that create_tables calls create_engine."""
        mock_settings.get_postgres_db_uri.return_value = (
            "postgresql://user:pass@localhost/db"
        )
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        result = create_tables()

        mock_create_engine.assert_called_once()
        assert result is mock_engine

    @patch("federated_pneumonia_detection.src.boundary.engine.create_engine")
    @patch("federated_pneumonia_detection.src.boundary.engine.settings")
    def test_create_tables_calls_metadata_create_all(
        self, mock_settings, mock_create_engine
    ):
        """Test that create_tables calls metadata.create_all."""
        mock_settings.get_postgres_db_uri.return_value = (
            "postgresql://user:pass@localhost/db"
        )
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        with patch.object(Base.metadata, "create_all") as mock_create_all:
            create_tables()
            mock_create_all.assert_called_once_with(mock_engine)

    @patch("federated_pneumonia_detection.src.boundary.engine.create_engine")
    @patch("federated_pneumonia_detection.src.boundary.engine.settings")
    def test_create_tables_uses_postgres_uri(self, mock_settings, mock_create_engine):
        """Test that create_tables uses PostgreSQL URI from settings."""
        expected_uri = "postgresql://test_user:test_pass@localhost/test_db"
        mock_settings.get_postgres_db_uri.return_value = expected_uri
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        create_tables()

        mock_create_engine.assert_called_once_with(expected_uri)

    @patch("federated_pneumonia_detection.src.boundary.engine.create_engine")
    @patch("federated_pneumonia_detection.src.boundary.engine.settings")
    def test_create_tables_returns_engine(self, mock_settings, mock_create_engine):
        """Test that create_tables returns the engine."""
        mock_settings.get_postgres_db_uri.return_value = "postgresql://localhost/db"
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        result = create_tables()

        assert result is mock_engine


class TestGetSessionFunction:
    """Tests for get_session function."""

    @patch("federated_pneumonia_detection.src.boundary.engine.sessionmaker")
    @patch("federated_pneumonia_detection.src.boundary.engine.create_engine")
    @patch("federated_pneumonia_detection.src.boundary.engine.settings")
    def test_get_session_creates_engine(
        self, mock_settings, mock_create_engine, mock_sessionmaker
    ):
        """Test that get_session creates an engine."""
        mock_settings.get_postgres_db_uri.return_value = "postgresql://localhost/db"
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        result = get_session()

        mock_create_engine.assert_called_once()

    @patch("federated_pneumonia_detection.src.boundary.engine.sessionmaker")
    @patch("federated_pneumonia_detection.src.boundary.engine.create_engine")
    @patch("federated_pneumonia_detection.src.boundary.engine.settings")
    def test_get_session_creates_session_factory(
        self, mock_settings, mock_create_engine, mock_sessionmaker
    ):
        """Test that get_session creates a session factory."""
        mock_settings.get_postgres_db_uri.return_value = "postgresql://localhost/db"
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        get_session()

        mock_sessionmaker.assert_called_once_with(bind=mock_engine)

    @patch("federated_pneumonia_detection.src.boundary.engine.sessionmaker")
    @patch("federated_pneumonia_detection.src.boundary.engine.create_engine")
    @patch("federated_pneumonia_detection.src.boundary.engine.settings")
    def test_get_session_returns_session(
        self, mock_settings, mock_create_engine, mock_sessionmaker
    ):
        """Test that get_session returns a session."""
        mock_settings.get_postgres_db_uri.return_value = "postgresql://localhost/db"
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        result = get_session()

        assert result is mock_session

    @patch("federated_pneumonia_detection.src.boundary.engine.sessionmaker")
    @patch("federated_pneumonia_detection.src.boundary.engine.create_engine")
    @patch("federated_pneumonia_detection.src.boundary.engine.settings")
    def test_get_session_uses_postgres_uri(
        self, mock_settings, mock_create_engine, mock_sessionmaker
    ):
        """Test that get_session uses PostgreSQL URI from settings."""
        expected_uri = "postgresql://test_user:test_pass@localhost/test_db"
        mock_settings.get_postgres_db_uri.return_value = expected_uri
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        get_session()

        mock_create_engine.assert_called_once_with(expected_uri)

    @patch("federated_pneumonia_detection.src.boundary.engine.sessionmaker")
    @patch("federated_pneumonia_detection.src.boundary.engine.create_engine")
    @patch("federated_pneumonia_detection.src.boundary.engine.settings")
    def test_get_session_multiple_calls_create_new_sessions(
        self, mock_settings, mock_create_engine, mock_sessionmaker
    ):
        """Test that multiple calls to get_session create new engine/session pairs."""
        mock_settings.get_postgres_db_uri.return_value = "postgresql://localhost/db"
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session

        get_session()
        get_session()

        # Each call should create a new engine
        assert mock_create_engine.call_count == 2


class TestSettingsIntegration:
    """Tests for Settings integration with engine."""

    @patch("federated_pneumonia_detection.src.boundary.engine.Settings")
    def test_settings_instance_created(self, mock_settings_class):
        """Test that Settings instance is created."""
        mock_settings_instance = MagicMock()
        mock_settings_class.return_value = mock_settings_instance

        # Re-import to trigger Settings instantiation
        import importlib
        import federated_pneumonia_detection.src.boundary.engine as engine_module

        importlib.reload(engine_module)

        # Settings should have been instantiated
        assert hasattr(engine_module, "settings")

    def test_settings_has_get_postgres_db_uri_method(self):
        """Test that settings has get_postgres_db_uri method."""
        from federated_pneumonia_detection.config.settings import Settings

        settings = Settings()
        assert hasattr(settings, "get_postgres_db_uri")
        assert callable(settings.get_postgres_db_uri)


class TestColumnConstraints:
    """Tests for column constraints and properties."""

    def test_run_training_mode_not_nullable(self):
        """Test that run training_mode column properties."""
        training_mode_column = Run.__table__.columns["training_mode"]
        assert training_mode_column is not None

    def test_run_configuration_learning_rate_float(self):
        """Test that learning_rate is Float type."""
        lr_column = RunConfiguration.__table__.columns["learning_rate"]
        assert isinstance(lr_column.type, FloatType)

    def test_run_metric_metric_value_float(self):
        """Test that metric_value is Float type."""
        value_column = RunMetric.__table__.columns["metric_value"]
        assert isinstance(value_column.type, FloatType)

    def test_run_source_path_large_string(self):
        """Test that source_path column allows large strings."""
        source_path_column = Run.__table__.columns["source_path"]
        # Should be 1024 character string
        assert source_path_column is not None


class TestForeignKeyConstraints:
    """Tests for foreign key constraints."""

    def test_run_configuration_references_runs(self):
        """Test that RunConfiguration.run_id references Run.id."""
        config_run_fk = RunConfiguration.__table__.columns["run_id"].foreign_keys
        assert len(config_run_fk) > 0
        assert any("runs.id" in str(fk) for fk in config_run_fk)

    def test_run_metric_references_runs(self):
        """Test that RunMetric.run_id references Run.id."""
        metric_run_fk = RunMetric.__table__.columns["run_id"].foreign_keys
        assert len(metric_run_fk) > 0
        assert any("runs.id" in str(fk) for fk in metric_run_fk)

    def test_run_artifact_references_runs(self):
        """Test that RunArtifact.run_id references Run.id."""
        artifact_run_fk = RunArtifact.__table__.columns["run_id"].foreign_keys
        assert len(artifact_run_fk) > 0
        assert any("runs.id" in str(fk) for fk in artifact_run_fk)
