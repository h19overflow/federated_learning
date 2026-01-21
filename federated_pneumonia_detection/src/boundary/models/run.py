"""
Run model - represents a training session (centralized or federated).
"""

from sqlalchemy import TIMESTAMP, Column, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Run(Base):
    """
    Represents a complete training execution (centralized or federated).

    Attributes:
        id: Unique run identifier
        run_description: Training description/notes
        training_mode: 'centralized' or 'federated'
        status: 'in_progress', 'completed', 'failed'
        start_time: Run start timestamp
        end_time: Run completion timestamp (NULL if in progress)
        wandb_id: Weights & Biases integration ID
        source_path: Dataset source path
    """

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    run_description = Column(String(1024), nullable=True)
    training_mode = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    start_time = Column(TIMESTAMP, nullable=False)
    end_time = Column(TIMESTAMP, nullable=True)
    wandb_id = Column(String(255), nullable=True)
    source_path = Column(String(1024), nullable=True)

    metrics = relationship("RunMetric", back_populates="run")
    clients = relationship("Client", back_populates="run")
    server_evaluations = relationship("ServerEvaluation", back_populates="run")
