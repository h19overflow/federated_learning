from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, TIMESTAMP, JSON, UniqueConstraint
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from federated_pneumonia_detection.config.settings import Settings

Base = declarative_base()
settings = Settings()

# TODO: in the federated trainer , client and the federated metrics collector,
#  we need to make sure we persist the data in the new format , as well as building the CRUD for the new models
# Make federated specific funtions in the CRUD layer for the run , run_metrics.
class Run(Base):
    __tablename__ = 'runs'

    id = Column(Integer, primary_key=True)
    run_description = Column(String(1024), nullable=True)
    training_mode = Column(String(50), nullable=False)  # 'centralized' or 'federated'
    status = Column(String(50), nullable=False)
    start_time = Column(TIMESTAMP, nullable=False)
    end_time = Column(TIMESTAMP, nullable=True)
    wandb_id = Column(String(255), nullable=True)
    source_path = Column(String(1024), nullable=True)

    configuration = relationship("RunConfiguration", back_populates="run", uselist=False)
    metrics = relationship("RunMetric", back_populates="run")
    artifacts = relationship("RunArtifact", back_populates="run")
    clients = relationship("Client", back_populates="run")


class Client(Base):
    __tablename__ = 'clients'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'), nullable=False)
    client_identifier = Column(String(255), nullable=False)  # 'client_0', 'client_1'
    created_at = Column(TIMESTAMP, nullable=False)

    # Optional: Store client-specific config as JSON for flexibility
    client_config = Column(JSON, nullable=True)  # Dataset split, learning rate overrides, etc.

    run = relationship("Run", back_populates="clients")
    rounds = relationship("Round", back_populates="client", cascade="all, delete-orphan")


class Round(Base):
    __tablename__ = 'rounds'

    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), nullable=False)
    round_number = Column(Integer, nullable=False)
    start_time = Column(TIMESTAMP, nullable=True)
    end_time = Column(TIMESTAMP, nullable=True)

    # Optional: Flexible metadata for aggregation strategy, sample weights, etc.
    round_metadata = Column(JSON, nullable=True)

    client = relationship("Client", back_populates="rounds")



class RunConfiguration(Base):
    __tablename__ = 'run_configurations'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'), nullable=False)

    # Shared configuration (used by both modes)
    learning_rate = Column(Float, nullable=False)
    epochs = Column(Integer, nullable=False)
    weight_decay = Column(Float, nullable=True)
    batch_size = Column(Integer, nullable=False)
    seed = Column(Integer, nullable=True)

    # Federated-specific configuration (NULL for centralized)
    num_rounds = Column(Integer, nullable=True)
    num_clients = Column(Integer, nullable=True)
    clients_per_round = Column(Integer, nullable=True)
    local_epochs = Column(Integer, nullable=True)
    partition_strategy = Column(String(50), nullable=True)

    run = relationship("Run", back_populates="configuration")

class RunMetric(Base):
    __tablename__ = 'run_metrics'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'), nullable=False)  # Always required

    # Federated-specific fields (NULL for centralized training)
    client_id = Column(Integer, ForeignKey('clients.id'), nullable=True)
    round_id = Column(Integer, ForeignKey('rounds.id'), nullable=True)

    # Core metric fields (used by both modes)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    step = Column(Integer, nullable=False)  # Epoch for centralized, local epoch for federated
    dataset_type = Column(String(50), nullable=True)  # 'train', 'val', 'test'

    # Optional: Add context field for flexibility
    context = Column(String(50), nullable=True)  # 'global', 'local', 'aggregated', or NULL

    run = relationship("Run", back_populates="metrics")
    client = relationship("Client")
    round = relationship("Round")


class RunArtifact(Base):
    __tablename__ = 'run_artifacts'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    artifact_name = Column(String(255))
    artifact_path = Column(String(1024))
    artifact_type = Column(String(50))
    
    run = relationship("Run", back_populates="artifacts")

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