"""
ServerEvaluation model - server-side evaluation metrics for federated learning.
"""

from sqlalchemy import JSON, TIMESTAMP, Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship

from .run import Base


class ServerEvaluation(Base):
    """
    Server-side evaluation metrics for federated learning.

    Stores centralized evaluation of the global model after each round.

    Attributes:
        id: Unique evaluation ID
        run_id: Parent federated run (foreign key)
        round_number: Federated round number
        loss: Validation loss
        accuracy: Classification accuracy
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        auroc: AUROC score
        true_positives: Confusion matrix TP
        true_negatives: Confusion matrix TN
        false_positives: Confusion matrix FP
        false_negatives: Confusion matrix FN
        num_samples: Samples in evaluation set
        evaluation_time: Evaluation timestamp
        additional_metrics: Additional metrics as JSON
    """

    __tablename__ = "server_evaluations"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    round_number = Column(Integer, nullable=False)

    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auroc = Column(Float, nullable=True)

    true_positives = Column(Integer, nullable=True)
    true_negatives = Column(Integer, nullable=True)
    false_positives = Column(Integer, nullable=True)
    false_negatives = Column(Integer, nullable=True)

    num_samples = Column(Integer, nullable=True)
    evaluation_time = Column(TIMESTAMP, nullable=False)

    additional_metrics = Column(JSON, nullable=True)

    run = relationship("Run", back_populates="server_evaluations")
