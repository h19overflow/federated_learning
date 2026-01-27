import json

from federated_pneumonia_detection.src.control.analytics.internals.services import (
    RankingService,
)


class TestRankingService:
    def test_top_runs_sorting_and_limit(self, mock_session, mock_cache, mock_run_crud):
        """Verify sorting by metric and limit n."""
        # Setup
        service = RankingService(cache=mock_cache, run_crud_obj=mock_run_crud)
        mock_cache.get.return_value = None  # Cache miss

        runs = [
            {"id": 1, "accuracy": 0.8},
            {"id": 2, "accuracy": 0.9},
            {"id": 3, "accuracy": 0.7},
        ]
        mock_run_crud.read_all.return_value = runs

        # Execute
        result = service.top_runs(mock_session, metric="accuracy", n=2)

        # Assert
        assert len(result) == 2
        assert result[0]["id"] == 2  # 0.9
        assert result[1]["id"] == 1  # 0.8
        mock_run_crud.read_all.assert_called_once_with(db=mock_session, filters={})

    def test_top_runs_cache_hit(self, mock_session, mock_cache, mock_run_crud):
        """Verify cache hit returns cached data and skips DB."""
        # Setup
        service = RankingService(cache=mock_cache, run_crud_obj=mock_run_crud)
        cached_data = [{"id": 1, "accuracy": 0.95}]
        mock_cache.get.return_value = json.dumps(cached_data)

        # Execute
        result = service.top_runs(mock_session, metric="accuracy", n=1)

        # Assert
        assert result == cached_data
        mock_run_crud.read_all.assert_not_called()
        mock_cache.get.assert_called_once()

    def test_top_runs_cache_miss_sets_cache(
        self, mock_session, mock_cache, mock_run_crud
    ):
        """Verify cache miss calls DB and sets cache."""
        # Setup
        service = RankingService(cache=mock_cache, run_crud_obj=mock_run_crud)
        mock_cache.get.return_value = None
        runs = [{"id": 1, "accuracy": 0.9}]
        mock_run_crud.read_all.return_value = runs

        # Execute
        result = service.top_runs(mock_session, metric="accuracy", n=1)

        # Assert
        assert result == runs
        mock_cache.set.assert_called_once()
        # Check that it was called with some key and the json-dumped result
        args, _ = mock_cache.set.call_args
        assert json.loads(args[1]) == runs

    def test_top_runs_empty_db(self, mock_session, mock_cache, mock_run_crud):
        """Verify behavior when DB returns no runs."""
        # Setup
        service = RankingService(cache=mock_cache, run_crud_obj=mock_run_crud)
        mock_cache.get.return_value = None
        mock_run_crud.read_all.return_value = []

        # Execute
        result = service.top_runs(mock_session, metric="accuracy", n=5)

        # Assert
        assert result == []
        mock_cache.set.assert_not_called()
