"""
Client model - represents a federated learning participant.
"""

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .base import Base


class Client(Base):
    """
    Represents a federated learning participant.

    Attributes:
        id: Unique client identifier
        run_id: Parent run ID (foreign key)
        client_identifier: Client label (e.g., 'client_0', 'client_1')
        created_at: Client registration timestamp
        client_config: Client-specific configuration as JSON (optional)
    """

    __tablename__ = "clients"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False, index=True)
    client_identifier = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)

    client_config = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint("run_id", "client_identifier", name="uq_run_client"),
        Index("ix_client_run_identifier", "run_id", "client_identifier"),
    )

    run = relationship("Run", back_populates="clients")
    rounds = relationship(
        "Round",
        back_populates="client",
        cascade="all, delete-orphan",
    )
