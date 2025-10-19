from sqlalchemy import create_engine, Column, Integer, String,  Float, ForeignKey, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from federated_pneumonia_detection.config.settings import Settings

Base = declarative_base()
settings = Settings()


class Run(Base):
    __tablename__ = 'runs'
    
    id = Column(Integer, primary_key=True)
    training_mode = Column(String(50))
    status = Column(String(50))
    start_time = Column(TIMESTAMP)
    end_time = Column(TIMESTAMP)
    wandb_id = Column(String(255))
    source_path = Column(String(1024))
    
    configuration = relationship("RunConfiguration", back_populates="run", uselist=False)
    metrics = relationship("RunMetric", back_populates="run")
    artifacts = relationship("RunArtifact", back_populates="run")

class RunConfiguration(Base):
    __tablename__ = 'run_configurations'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    learning_rate = Column(Float)
    epochs = Column(Integer)
    weight_decay = Column(Float)
    batch_size = Column(Integer)
    num_rounds = Column(Integer)
    num_clients = Column(Integer)
    clients_per_round = Column(Integer)
    local_epochs = Column(Integer)
    partition_strategy = Column(String(50))
    seed = Column(Integer)
    
    run = relationship("Run", back_populates="configuration")

class RunMetric(Base):
    __tablename__ = 'run_metrics'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    metric_name = Column(String(255))
    metric_value = Column(Float)
    step = Column(Integer)
    dataset_type = Column(String(50))
    
    run = relationship("Run", back_populates="metrics")

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