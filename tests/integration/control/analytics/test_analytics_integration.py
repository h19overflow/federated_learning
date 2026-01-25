import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from federated_pneumonia_detection.src.boundary.models.base import Base
from federated_pneumonia_detection.src.boundary.models import Run, RunMetric
from federated_pneumonia_detection.src.control.analytics.facade import AnalyticsFacade
from federated_pneumonia_detection.src.control.analytics.internals import (
    SummaryService,
    MetricsService,
    CacheProvider,
)


def test_full_analytics_flow_centralized():
    # Setup Logic
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Arrange
        now = datetime.now()
        run = Run(
            training_mode="centralized",
            status="completed",
            start_time=now,
            run_description="Test Run",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        # Create 2 RunMetric objects linked to this run (epoch 1: acc=0.8, epoch 2: acc=0.9)
        metric1 = RunMetric(
            run_id=run.id, metric_name="val_acc", metric_value=0.8, step=1
        )
        metric2 = RunMetric(
            run_id=run.id, metric_name="val_acc", metric_value=0.9, step=2
        )
        session.add_all([metric1, metric2])
        session.commit()

        # Act
        cache = CacheProvider(ttl=600)
        summary_service = SummaryService(cache=cache)
        metrics_service = MetricsService(cache=cache)

        # Instantiate AnalyticsFacade(session=session)
        facade = AnalyticsFacade(summary=summary_service, metrics=metrics_service)

        # Call result = facade.get_run_summary(run.id)
        result = facade.get_run_summary(session, run.id)

        # Assert
        # result.metrics['best_accuracy'] is close to 0.9.
        assert result.metrics["best_accuracy"] == pytest.approx(0.9)
        # result.status == 'completed'.
        assert result.status == "completed"
        # result.mode == 'centralized'.
        assert result.mode == "centralized"

    finally:
        session.close()
        engine.dispose()
