"""Unit tests for client metrics functionality in MetricsService."""

from unittest.mock import MagicMock

import pytest

from federated_pneumonia_detection.src.control.analytics.internals.services import (
    MetricsService,
)


class TestClientMetrics:
    """Test suite for get_client_metrics and related methods."""

    @pytest.fixture
    def mock_run_metric_crud(self):
        """Mock RunMetricCRUD with client metrics methods."""
        crud = MagicMock()
        crud.get_by_run_grouped_by_client.return_value = {}
        crud.get_aggregated_metrics_by_run.return_value = []
        return crud

    @pytest.fixture
    def mock_client_crud(self):
        """Mock ClientCRUD."""
        crud = MagicMock()
        crud.get_clients_by_run.return_value = []
        return crud

    @pytest.fixture
    def service(
        self, mock_cache, mock_run_crud, mock_run_metric_crud, mock_client_crud
    ):
        """Create MetricsService with mocked dependencies."""
        return MetricsService(
            cache=mock_cache,
            run_crud_obj=mock_run_crud,
            run_metric_crud_obj=mock_run_metric_crud,
            client_crud_obj=mock_client_crud,
        )

    def test_get_client_metrics_centralized_run(
        self, service, mock_session, mock_run_crud
    ):
        """Test that centralized runs return is_federated=False."""
        # Setup
        run = MagicMock()
        run.id = 1
        run.training_mode = "centralized"
        mock_run_crud.get_with_metrics.return_value = run

        # Execute
        result = service.get_client_metrics(mock_session, 1)

        # Assert
        assert result["run_id"] == 1
        assert result["is_federated"] is False
        assert result["clients"] == []
        assert result["aggregated_metrics"] == []

    def test_get_client_metrics_federated_no_clients(
        self, service, mock_session, mock_run_crud, mock_client_crud
    ):
        """Test federated run with no clients returns empty structure."""
        # Setup
        run = MagicMock()
        run.id = 2
        run.training_mode = "federated"
        mock_run_crud.get_with_metrics.return_value = run
        mock_client_crud.get_clients_by_run.return_value = []

        # Execute
        result = service.get_client_metrics(mock_session, 2)

        # Assert
        assert result["run_id"] == 2
        assert result["is_federated"] is True
        assert result["num_clients"] == 0
        assert result["clients"] == []

    def test_get_client_metrics_with_clients(
        self,
        service,
        mock_session,
        mock_run_crud,
        mock_client_crud,
        mock_run_metric_crud,
    ):
        """Test federated run with clients returns properly grouped data."""
        # Setup run
        run = MagicMock()
        run.id = 3
        run.training_mode = "federated"
        mock_run_crud.get_with_metrics.return_value = run

        # Setup clients
        client1 = MagicMock()
        client1.id = 100
        client1.client_identifier = "client_0"
        client2 = MagicMock()
        client2.id = 101
        client2.client_identifier = "client_1"
        mock_client_crud.get_clients_by_run.return_value = [client1, client2]

        # Setup metrics for client 1
        metric1 = MagicMock()
        metric1.client_id = 100
        metric1.step = 0
        metric1.metric_name = "val_acc"
        metric1.metric_value = 0.75
        metric1.round = MagicMock()
        metric1.round.round_number = 1

        metric2 = MagicMock()
        metric2.client_id = 100
        metric2.step = 1
        metric2.metric_name = "val_acc"
        metric2.metric_value = 0.80
        metric2.round = MagicMock()
        metric2.round.round_number = 1

        # Setup metrics for client 2
        metric3 = MagicMock()
        metric3.client_id = 101
        metric3.step = 0
        metric3.metric_name = "val_acc"
        metric3.metric_value = 0.70
        metric3.round = MagicMock()
        metric3.round.round_number = 1

        mock_run_metric_crud.get_by_run_grouped_by_client.return_value = {
            100: [metric1, metric2],
            101: [metric3],
        }
        mock_run_metric_crud.get_aggregated_metrics_by_run.return_value = []

        # Execute
        result = service.get_client_metrics(mock_session, 3)

        # Assert
        assert result["run_id"] == 3
        assert result["is_federated"] is True
        assert result["num_clients"] == 2
        assert len(result["clients"]) == 2

        # Verify client_0 data
        client_0_data = next(
            c for c in result["clients"] if c["client_identifier"] == "client_0"
        )
        assert client_0_data["client_id"] == 100
        assert client_0_data["total_steps"] == 2
        assert client_0_data["best_metrics"]["best_val_accuracy"] == 0.80

        # Verify client_1 data
        client_1_data = next(
            c for c in result["clients"] if c["client_identifier"] == "client_1"
        )
        assert client_1_data["client_id"] == 101
        assert client_1_data["total_steps"] == 1
        assert client_1_data["best_metrics"]["best_val_accuracy"] == 0.70

    def test_get_client_metrics_with_aggregated(
        self,
        service,
        mock_session,
        mock_run_crud,
        mock_client_crud,
        mock_run_metric_crud,
    ):
        """Test federated run returns aggregated metrics."""
        # Setup run
        run = MagicMock()
        run.id = 4
        run.training_mode = "federated"
        mock_run_crud.get_with_metrics.return_value = run
        mock_client_crud.get_clients_by_run.return_value = []
        mock_run_metric_crud.get_by_run_grouped_by_client.return_value = {}

        # Setup aggregated metrics
        agg1 = MagicMock()
        agg1.step = 0
        agg1.metric_name = "val_accuracy"
        agg1.metric_value = 0.78

        agg2 = MagicMock()
        agg2.step = 1
        agg2.metric_name = "val_accuracy"
        agg2.metric_value = 0.82

        agg3 = MagicMock()
        agg3.step = 1
        agg3.metric_name = "val_f1"
        agg3.metric_value = 0.80

        mock_run_metric_crud.get_aggregated_metrics_by_run.return_value = [
            agg1,
            agg2,
            agg3,
        ]

        # Execute
        result = service.get_client_metrics(mock_session, 4)

        # Assert
        assert len(result["aggregated_metrics"]) == 2
        assert result["aggregated_metrics"][0]["round"] == 0
        assert result["aggregated_metrics"][0]["val_accuracy"] == 0.78
        assert result["aggregated_metrics"][1]["round"] == 1
        assert result["aggregated_metrics"][1]["val_accuracy"] == 0.82
        assert result["aggregated_metrics"][1]["val_f1"] == 0.80

    def test_get_client_metrics_run_not_found(self, service, mock_session, mock_run_crud):
        """Test that ValueError is raised for non-existent run."""
        mock_run_crud.get_with_metrics.return_value = None

        with pytest.raises(ValueError, match="Run 999 not found"):
            service.get_client_metrics(mock_session, 999)


class TestTransformClientMetrics:
    """Test suite for _transform_client_metrics helper method."""

    @pytest.fixture
    def service(self, mock_cache):
        """Create MetricsService with minimal mocks."""
        return MetricsService(cache=mock_cache)

    def test_transform_empty_metrics(self, service):
        """Test transformation with no metrics returns empty history."""
        result = service._transform_client_metrics(1, "client_0", [])

        assert result["client_id"] == 1
        assert result["client_identifier"] == "client_0"
        assert result["total_steps"] == 0
        assert result["training_history"] == []

    def test_transform_multiple_metrics_per_step(self, service):
        """Test that metrics at same step are grouped together."""
        metric1 = MagicMock()
        metric1.step = 0
        metric1.metric_name = "val_acc"
        metric1.metric_value = 0.75
        metric1.round = None

        metric2 = MagicMock()
        metric2.step = 0
        metric2.metric_name = "val_loss"
        metric2.metric_value = 0.25
        metric2.round = None

        result = service._transform_client_metrics(1, "client_0", [metric1, metric2])

        assert result["total_steps"] == 1
        assert len(result["training_history"]) == 1
        assert result["training_history"][0]["step"] == 0
        assert result["training_history"][0]["val_acc"] == 0.75
        assert result["training_history"][0]["val_loss"] == 0.25


class TestCalculateClientBestMetrics:
    """Test suite for _calculate_client_best_metrics helper method."""

    @pytest.fixture
    def service(self, mock_cache):
        """Create MetricsService with minimal mocks."""
        return MetricsService(cache=mock_cache)

    def test_empty_history(self, service):
        """Test that empty history returns None for all metrics."""
        result = service._calculate_client_best_metrics([])

        assert result["best_val_accuracy"] is None
        assert result["best_val_precision"] is None
        assert result["best_val_recall"] is None
        assert result["best_val_f1"] is None
        assert result["best_val_auroc"] is None
        assert result["lowest_val_loss"] is None

    def test_best_metrics_selected(self, service):
        """Test that best values are correctly selected."""
        history = [
            {"step": 0, "val_acc": 0.6, "val_loss": 0.5},
            {"step": 1, "val_acc": 0.8, "val_loss": 0.3},
            {"step": 2, "val_acc": 0.7, "val_loss": 0.4},
        ]

        result = service._calculate_client_best_metrics(history)

        assert result["best_val_accuracy"] == 0.8
        assert result["lowest_val_loss"] == 0.3
