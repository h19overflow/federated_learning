from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.CRUD.run import RunCRUD, run_crud
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import (
    RunMetricCRUD,
    run_metric_crud,
)
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    ServerEvaluationCRUD,
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.boundary.CRUD.client import ClientCRUD
from federated_pneumonia_detection.src.boundary.CRUD.round import RoundCRUD, round_crud
from federated_pneumonia_detection.src.boundary.CRUD.fetch_documents import fetch_all_documents

client_crud = ClientCRUD()

__all__ = [
    "BaseCRUD",
    "RunCRUD",
    "RunMetricCRUD",
    "ServerEvaluationCRUD",
    "ClientCRUD",
    "RoundCRUD",
    "run_crud",
    "run_metric_crud",
    "server_evaluation_crud",
    "client_crud",
    "round_crud",
    "fetch_all_documents",
]
