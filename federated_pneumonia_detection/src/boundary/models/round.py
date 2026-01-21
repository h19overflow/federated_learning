"""
Round model - tracks federated learning communication rounds per client.
"""

from sqlalchemy import Column, Integer, ForeignKey, TIMESTAMP, JSON
from sqlalchemy.orm import relationship
from .run import Base


class Round(Base):
    """
    Tracks federated learning communication rounds per client.

    Attributes:
        id: Unique round identifier
        client_id: Client participating in round (foreign key)
        round_number: Round number (1, 2, 3, ...)
        start_time: Round start timestamp
        end_time: Round completion timestamp
        round_metadata: Flexible metadata as JSON (aggregation strategy, weights, etc.)
    """

    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    start_time = Column(TIMESTAMP, nullable=True)
    end_time = Column(TIMESTAMP, nullable=True)

    round_metadata = Column(JSON, nullable=True)

    client = relationship("Client", back_populates="rounds")
