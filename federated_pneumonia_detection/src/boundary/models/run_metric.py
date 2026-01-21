"""
RunMetric model - stores training metrics per epoch/round.
"""

from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .run import Base


class RunMetric(Base):
    """
    Stores training metrics (loss, accuracy, precision, etc.) per epoch/round.

    Supports both centralized training (client_id/round_id are NULL) and
    federated training (client_id/round_id are set).

    Attributes:
        id: Unique metric record ID
        run_id: Parent run ID (foreign key, always required)
        client_id: Client ID (foreign key, NULL for centralized)
        round_id: Round ID (foreign key, NULL for centralized)
        metric_name: Metric name (e.g., 'val_loss', 'train_accuracy')
        metric_value: Metric value
        step: Epoch or step number
        dataset_type: 'train', 'val', 'test'
        context: 'global', 'local', 'aggregated', or NULL
    """

    __tablename__ = "run_metrics"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)

    client_id = Column(Integer, ForeignKey("clients.id"), nullable=True)
    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=True)

    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    step = Column(Integer, nullable=False)
    dataset_type = Column(String(50), nullable=True)

    context = Column(String(50), nullable=True)

    run = relationship("Run", back_populates="metrics")
    client = relationship("Client")
    round = relationship("Round")
