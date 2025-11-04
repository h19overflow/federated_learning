from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    TIMESTAMP,
    JSON,
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from federated_pneumonia_detection.config.settings import Settings

Base = declarative_base()
settings = Settings()


# TODO: in the federated trainer , client and the federated metrics collector,
#  we need to make sure we persist the data in the new format , as well as building the CRUD for the new models
# Make federated specific funtions in the CRUD layer for the run , run_metrics.
class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    run_description = Column(String(1024), nullable=True)
    training_mode = Column(String(50), nullable=False)  # 'centralized' or 'federated'
    status = Column(String(50), nullable=False)
    start_time = Column(TIMESTAMP, nullable=False)
    end_time = Column(TIMESTAMP, nullable=True)
    wandb_id = Column(String(255), nullable=True)
    source_path = Column(String(1024), nullable=True)

    metrics = relationship("RunMetric", back_populates="run")
    clients = relationship("Client", back_populates="run")
    server_evaluations = relationship("ServerEvaluation", back_populates="run")


class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    client_identifier = Column(String(255), nullable=False)  # 'client_0', 'client_1'
    created_at = Column(TIMESTAMP, nullable=False)

    # Optional: Store client-specific config as JSON for flexibility
    client_config = Column(
        JSON, nullable=True
    )  # Dataset split, learning rate overrides, etc.

    run = relationship("Run", back_populates="clients")
    rounds = relationship(
        "Round", back_populates="client", cascade="all, delete-orphan"
    )


class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    start_time = Column(TIMESTAMP, nullable=True)
    end_time = Column(TIMESTAMP, nullable=True)

    # Optional: Flexible metadata for aggregation strategy, sample weights, etc.
    round_metadata = Column(JSON, nullable=True)

    client = relationship("Client", back_populates="rounds")


class RunMetric(Base):
    __tablename__ = "run_metrics"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)  # Always required

    # Federated-specific fields (NULL for centralized training)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=True)
    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=True)

    # Core metric fields (used by both modes)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    step = Column(
        Integer, nullable=False
    )  # Epoch for centralized, local epoch for federated
    dataset_type = Column(String(50), nullable=True)  # 'train', 'val', 'test'

    # Optional: Add context field for flexibility
    context = Column(
        String(50), nullable=True
    )  # 'global', 'local', 'aggregated', or NULL

    run = relationship("Run", back_populates="metrics")
    client = relationship("Client")
    round = relationship("Round")


class ServerEvaluation(Base):
    """
    Server-side evaluation metrics for federated learning.
    Stores centralized evaluation of the global model after each round.
    """

    __tablename__ = "server_evaluations"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    round_number = Column(Integer, nullable=False)

    # Core evaluation metrics
    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auroc = Column(Float, nullable=True)

    # Confusion matrix components
    true_positives = Column(Integer, nullable=True)
    true_negatives = Column(Integer, nullable=True)
    false_positives = Column(Integer, nullable=True)
    false_negatives = Column(Integer, nullable=True)

    # Additional metadata
    num_samples = Column(Integer, nullable=True)
    evaluation_time = Column(TIMESTAMP, nullable=False)

    # Store any additional metrics as JSON
    additional_metrics = Column(JSON, nullable=True)

    run = relationship("Run", back_populates="server_evaluations")


def create_tables():
    engine = create_engine(settings.get_postgres_db_uri())
    Base.metadata.create_all(engine)
    return engine


def get_session():
    engine = create_engine(settings.get_postgres_db_uri())
    Session = sessionmaker(bind=engine)
    return Session()


def get_engine():
    engine = create_engine(settings.get_postgres_db_uri())
    return engine


if __name__ == "__main__":
    create_tables()
