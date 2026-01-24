"""Base class for metric extraction strategies."""

from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy.orm import Session


class MetricExtractor(ABC):
    """Base class for extracting metrics from different training modes."""

    @abstractmethod
    def get_best_metric(
        self,
        db: Session,
        run_id: int,
        metric_name: str,
    ) -> Optional[float]:
        """Extract best metric value for a run."""
        pass
