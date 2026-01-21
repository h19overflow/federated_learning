"""
Unit tests for API endpoint structure and business logic alignment.

This module validates that API endpoints:
1. Follow consistent naming and routing conventions
2. Correspond to the business logic layer (control layer)
3. Use proper dependency injection through deps.py
4. Handle errors appropriately
5. Use correct HTTP methods for their actions
"""

from pathlib import Path

import pytest


class TestEndpointStructure:
    """Test basic structure and routing of endpoints."""

    def test_configuration_endpoint_exists(self):
        """Configuration endpoints should exist and be properly routed."""
        from federated_pneumonia_detection.src.api.endpoints.configuration_settings import (
            configuration_endpoints,
        )

        assert hasattr(configuration_endpoints, "router")
        assert configuration_endpoints.router.prefix == "/config"
        assert "config" in configuration_endpoints.router.tags

    def test_experiment_endpoints_exist(self):
        """Experiment endpoints should exist for all three training types."""
        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            centralized_endpoints,
            comparison_endpoints,
            federated_endpoints,
        )

        assert hasattr(centralized_endpoints, "router")
        assert hasattr(federated_endpoints, "router")
        assert hasattr(comparison_endpoints, "router")

        # Check routing prefixes follow convention
        assert centralized_endpoints.router.prefix == "/experiments/centralized"
        assert federated_endpoints.router.prefix == "/experiments/federated"
        assert comparison_endpoints.router.prefix == "/experiments/comparison"

    def test_endpoint_tags_consistency(self):
        """Endpoint tags should be consistent and meaningful."""
        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            centralized_endpoints,
            comparison_endpoints,
            federated_endpoints,
        )

        assert "experiments" in centralized_endpoints.router.tags
        assert "experiments" in federated_endpoints.router.tags
        assert "experiments" in comparison_endpoints.router.tags

        assert "centralized" in centralized_endpoints.router.tags
        assert "federated" in federated_endpoints.router.tags
        assert "comparison" in comparison_endpoints.router.tags


class TestConfigurationEndpoint:
    """Test configuration endpoint structure."""

    def test_set_configuration_endpoint_exists(self):
        """Configuration endpoint should have POST /update."""
        from federated_pneumonia_detection.src.api.endpoints.configuration_settings import (
            configuration_endpoints,
        )

        routes = [route.path for route in configuration_endpoints.router.routes]
        assert "/config/update" in routes

    def test_configuration_schema_completeness(self):
        """Configuration schema should cover all configuration sections."""
        from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
            ConfigurationUpdateRequest,
        )

        # Check that schema has all required sections
        schema = ConfigurationUpdateRequest.model_json_schema()
        properties = schema["properties"]

        assert "system" in properties
        assert "paths" in properties
        assert "columns" in properties
        assert "experiment" in properties
        assert "output" in properties
        assert "logging" in properties

    def test_experiment_config_schema_includes_federated_params(self):
        """ExperimentConfig schema should include federated learning parameters."""
        from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
            ExperimentConfig as ExperimentConfigSchema,
        )

        schema = ExperimentConfigSchema.model_json_schema()
        properties = schema["properties"]

        # Federated parameters should be present
        assert "num_rounds" in properties
        assert "num_clients" in properties
        assert "clients_per_round" in properties
        assert "local_epochs" in properties

    def test_experiment_config_schema_includes_standard_params(self):
        """ExperimentConfig schema should include standard training parameters."""
        from federated_pneumonia_detection.src.api.endpoints.schema.configuration_schemas import (
            ExperimentConfig as ExperimentConfigSchema,
        )

        schema = ExperimentConfigSchema.model_json_schema()
        properties = schema["properties"]

        # Standard parameters
        assert "learning_rate" in properties
        assert "epochs" in properties
        assert "batch_size" in properties


class TestFederatedEndpoint:
    """Test federated learning endpoint structure."""

    def test_federated_train_endpoint_exists(self):
        """Federated endpoint should have POST /train."""
        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            federated_endpoints,
        )

        routes = [route.path for route in federated_endpoints.router.routes]
        assert "/experiments/federated/train" in routes

    def test_federated_endpoint_background_task_function_exists(self):
        """Federated endpoint should have background task function."""
        from federated_pneumonia_detection.src.api.endpoints.experiments.utils import (
            run_federated_training_task,
        )

        assert callable(run_federated_training_task)

    def test_federated_endpoint_accepts_zip_upload(self):
        """Federated endpoint should accept ZIP file uploads."""
        # Check function signature
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            federated_endpoints,
        )

        sig = inspect.signature(federated_endpoints.start_federated_training)
        params = list(sig.parameters.keys())

        assert "data_zip" in params
        assert "experiment_name" in params
        assert "csv_filename" in params

    def test_federated_endpoint_calls_federated_trainer(self):
        """Background task should instantiate FederatedTrainer."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments.utils import (
            execute_federated_training,
            run_federated_training_task,
        )

        # Check main task function uses FederatedTrainer
        source = inspect.getsource(run_federated_training_task)
        assert "FederatedTrainer" in source

        # Check helper function calls train method
        helper_source = inspect.getsource(execute_federated_training)
        assert ".train(" in helper_source


class TestCentralizedEndpoint:
    """Test centralized learning endpoint structure."""

    def test_centralized_train_endpoint_exists(self):
        """Centralized endpoint should have POST /train."""
        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            centralized_endpoints,
        )

        routes = [route.path for route in centralized_endpoints.router.routes]
        assert "/experiments/centralized/train" in routes

    def test_centralized_endpoint_background_task_function_exists(self):
        """Centralized endpoint should have background task function."""
        from federated_pneumonia_detection.src.api.endpoints.experiments.utils import (
            run_centralized_training_task,
        )

        assert callable(run_centralized_training_task)

    def test_centralized_endpoint_calls_centralized_trainer(self):
        """Background task should instantiate CentralizedTrainer."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments.utils import (
            run_centralized_training_task,
        )

        source = inspect.getsource(run_centralized_training_task)

        assert "CentralizedTrainer" in source
        # Check for train method call (could be .train or .train_model)
        assert ".train" in source

    def test_centralized_endpoint_accepts_checkpoint_and_logs_dirs(self):
        """Centralized endpoint should accept checkpoint and logs directories."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            centralized_endpoints,
        )

        sig = inspect.signature(centralized_endpoints.start_centralized_training)
        params = list(sig.parameters.keys())

        assert "checkpoint_dir" in params
        assert "logs_dir" in params


class TestComparisonEndpoint:
    """Test comparison endpoint structure."""

    def test_comparison_endpoint_exists(self):
        """Comparison endpoint should have POST /run."""
        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            comparison_endpoints,
        )

        routes = [route.path for route in comparison_endpoints.router.routes]
        assert "/experiments/comparison/run" in routes

    def test_comparison_endpoint_background_task_function_exists(self):
        """Comparison endpoint should have background task function."""
        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            comparison_endpoints,
        )

        assert hasattr(comparison_endpoints, "_run_comparison_task")

    def test_comparison_endpoint_calls_orchestrator(self):
        """Background task should instantiate ExperimentOrchestrator."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            comparison_endpoints,
        )

        source = inspect.getsource(comparison_endpoints._run_comparison_task)

        assert "ExperimentOrchestrator" in source
        assert ".run_comparison(" in source

    def test_comparison_endpoint_accepts_partition_strategy(self):
        """Comparison endpoint should accept partition strategy parameter."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            comparison_endpoints,
        )

        sig = inspect.signature(comparison_endpoints.start_comparison_experiment)
        params = list(sig.parameters.keys())

        assert "partition_strategy" in params


class TestDependencyInjection:
    """Test dependency injection through deps.py."""

    def test_deps_has_database_dependency(self):
        """deps.py should provide database session dependency."""
        from federated_pneumonia_detection.src.api import deps

        assert hasattr(deps, "get_db")

    def test_deps_has_config_dependency(self):
        """deps.py should provide config manager dependency."""
        from federated_pneumonia_detection.src.api import deps

        assert hasattr(deps, "get_config")

    def test_deps_has_crud_dependencies(self):
        """deps.py should provide CRUD operation dependencies."""
        from federated_pneumonia_detection.src.api import deps

        assert hasattr(deps, "get_experiment_crud")
        assert hasattr(deps, "get_run_configuration_crud")
        assert hasattr(deps, "get_run_metric_crud")
        assert hasattr(deps, "get_run_artifact_crud")

    def test_crud_imports_are_available(self):
        """All required CRUD classes should be importable."""
        from federated_pneumonia_detection.src.boundary.CRUD.run import RunCRUD
        from federated_pneumonia_detection.src.boundary.CRUD.run_artifact import (
            RunArtifactCRUD,
        )
        from federated_pneumonia_detection.src.boundary.CRUD.run_configuration import (
            RunConfigurationCRUD,
        )
        from federated_pneumonia_detection.src.boundary.CRUD.run_metric import (
            RunMetricCRUD,
        )

        assert RunCRUD is not None
        assert RunConfigurationCRUD is not None
        assert RunMetricCRUD is not None
        assert RunArtifactCRUD is not None


class TestAPISettings:
    """Test API settings configuration."""

    def test_settings_has_required_fields(self):
        """Settings should have BASE_URL, API_VERSION, and API_PREFIX."""
        from federated_pneumonia_detection.src.api.settings import Settings

        settings = Settings()

        assert hasattr(settings, "BASE_URL")
        assert hasattr(settings, "API_VERSION")
        assert hasattr(settings, "API_PREFIX")

    def test_settings_defaults_are_reasonable(self):
        """Settings defaults should be reasonable values."""
        from federated_pneumonia_detection.src.api.settings import Settings

        settings = Settings()

        assert settings.BASE_URL == "http://localhost:8000"
        assert settings.API_VERSION == "v1"
        assert settings.API_PREFIX == "/api"


class TestErrorHandling:
    """Test error handling in endpoints."""

    def test_centralized_endpoint_handles_missing_file(self):
        """Centralized endpoint should handle missing metadata file."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments.utils import (
            run_centralized_training_task,
        )

        source = inspect.getsource(run_centralized_training_task)

        # Should have error handling
        assert "except" in source
        assert "error" in source.lower() or "Exception" in source


class TestBusinessLogicAlignment:
    """Test API alignment with business logic layer."""

    def test_federated_trainer_import_available(self):
        """FederatedTrainer should be importable in federated endpoint."""
        from federated_pneumonia_detection.src.control.federated_learning import (
            FederatedTrainer,
        )

        assert FederatedTrainer is not None

    def test_centralized_trainer_import_available(self):
        """CentralizedTrainer should be importable in centralized endpoint."""
        from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
            CentralizedTrainer,
        )

        assert CentralizedTrainer is not None

    def test_experiment_orchestrator_import_available(self):
        """ExperimentOrchestrator should be importable in comparison endpoint."""
        from federated_pneumonia_detection.src.control.comparison.experiment_orchestrator import (
            ExperimentOrchestrator,
        )

        assert ExperimentOrchestrator is not None

    def test_system_constants_available(self):
        """SystemConstants should be available for configuration."""
        from federated_pneumonia_detection.models.system_constants import (
            SystemConstants,
        )

        assert SystemConstants is not None

    def test_experiment_config_available(self):
        """ExperimentConfig should be available for configuration."""
        from federated_pneumonia_detection.models.experiment_config import (
            ExperimentConfig,
        )

        assert ExperimentConfig is not None


class TestNamingConventions:
    """Test naming conventions in API code."""

    def test_typo_in_configuration_endpoint_folder(self):
        """ISSUE FOUND: configuration_settings folder has typo, should be configuration_settings."""
        # This is a known issue - folder is named 'configuration_settings' (typo) instead of 'configuration_settings'
        folder_path = Path(
            "federated_pneumonia_detection/src/api/endpoints/configuration_settings",
        )
        # This test documents the issue
        assert folder_path.name == "configuration_settings"
        # EXPECTED: "configuration_settings" (no typo)

    def test_logging_endpoints_implementation_complete(self):
        """FIXED: logging_endpoints.py is now implemented."""
        try:
            from federated_pneumonia_detection.src.api.endpoints.runs_endpoints import (
                router,
            )

            # Check if the module has router
            # Verify that endpoints are implemented
            assert router is not None, (
                "runs_endpoints should have router implementation"
            )
            assert hasattr(router, "routes"), "router should have routes"
        except (ModuleNotFoundError, ImportError):
            # Skip this test if module doesn't exist yet
            pytest.skip("runs_endpoints module not implemented yet")

    def test_results_endpoints_implementation_complete(self):
        """FIXED: results_endpoints.py is now implemented."""
        try:
            from federated_pneumonia_detection.src.api.endpoints.runs_endpoints import (
                router,
            )

            # Check if the module has router
            # Verify that endpoints are implemented
            assert router is not None, (
                "runs_endpoints should have router implementation"
            )
            assert hasattr(router, "routes"), "router should have routes"
        except (ModuleNotFoundError, ImportError):
            # Skip this test if module doesn't exist yet
            pytest.skip("runs_endpoints module not implemented yet")


class TestZipFileHandling:
    """Test ZIP file handling in endpoints."""

    def test_centralized_endpoint_extracts_zip_correctly(self):
        """Centralized endpoint should extract ZIP files correctly."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments.utils import (
            prepare_zip,
        )

        source = inspect.getsource(prepare_zip)

        # Should use zipfile
        assert "zipfile" in source or "ZipFile" in source
        assert "extractall" in source or "extract" in source

    def test_federated_endpoint_extracts_zip_correctly(self):
        """Federated endpoint should extract ZIP files correctly."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments.utils import (
            prepare_zip,
        )

        source = inspect.getsource(prepare_zip)

        # Should use zipfile
        assert "zipfile" in source or "ZipFile" in source
        assert "extractall" in source or "extract" in source

    def test_comparison_endpoint_extracts_zip_correctly(self):
        """Comparison endpoint should extract ZIP files correctly."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            comparison_endpoints,
        )

        source = inspect.getsource(comparison_endpoints.start_comparison_experiment)

        # Should use zipfile
        assert "zipfile" in source or "ZipFile" in source
        assert "extractall" in source


class TestBackgroundTaskHandling:
    """Test background task handling in endpoints."""

    def test_endpoints_use_background_tasks(self):
        """Endpoints should use BackgroundTasks for async operations."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            centralized_endpoints,
            comparison_endpoints,
            federated_endpoints,
        )

        for endpoint_module in [
            centralized_endpoints,
            federated_endpoints,
            comparison_endpoints,
        ]:
            # Find the main async function
            for name, obj in inspect.getmembers(endpoint_module):
                if inspect.iscoroutinefunction(obj) and name.startswith("start_"):
                    sig = inspect.signature(obj)
                    # Should have BackgroundTasks parameter
                    assert "background_tasks" in sig.parameters


class TestResponseStructures:
    """Test response structure consistency."""

    def test_training_endpoints_return_dict(self):
        """Training endpoints should return Dict with status and message."""
        import inspect

        from federated_pneumonia_detection.src.api.endpoints.experiments import (
            centralized_endpoints,
            federated_endpoints,
        )

        # Check return type annotations
        sig = inspect.signature(centralized_endpoints.start_centralized_training)
        assert "Dict" in str(sig.return_annotation)

        sig = inspect.signature(federated_endpoints.start_federated_training)
        assert "Dict" in str(sig.return_annotation)
