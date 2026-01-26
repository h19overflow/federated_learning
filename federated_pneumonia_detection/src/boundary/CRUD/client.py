import datetime
from typing import List, Optional

from sqlalchemy.exc import SQLAlchemyError

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.models import Client


class ClientCRUD:
    def create_client(
        self,
        run_id: int,
        client_identifier: str,
        client_config: Optional[dict] = None,
    ):
        """Create a new client"""
        with get_session() as session:
            try:
                new_client = Client(
                    run_id=run_id,
                    client_identifier=client_identifier,
                    created_at=datetime.datetime.utcnow(),
                    client_config=client_config,
                )
                session.add(new_client)
                session.flush()
                session.refresh(new_client)
                session.commit()
                session.expunge(new_client)
                return new_client
            except SQLAlchemyError:
                session.rollback()
                raise

    def get_client_by_id(self, client_id: int) -> Optional[Client]:
        """Get a specific client by ID"""
        with get_session() as session:
            client_instance = (
                session.query(Client).filter(Client.id == client_id).first()
            )
            if client_instance:
                session.expunge(client_instance)
            return client_instance

    def get_client_by_identifier(
        self,
        run_id: int,
        client_identifier: str,
    ) -> Optional[Client]:
        """Get a specific client by run_id and client_identifier"""
        with get_session() as session:
            client_instance = (
                session.query(Client)
                .filter(
                    Client.run_id == run_id,
                    Client.client_identifier == client_identifier,
                )
                .first()
            )
            if client_instance:
                session.expunge(client_instance)
            return client_instance

    def get_clients_by_run_id(self, run_id: int) -> List[Client]:
        """Get all clients for a specific run"""
        with get_session() as session:
            clients = session.query(Client).filter(Client.run_id == run_id).all()
            for client_instance in clients:
                session.expunge(client_instance)
            return clients

    def set_client_config(self, client_id: int, configs: dict):
        """Set client configurations"""
        with get_session() as session:
            try:
                client_instance = (
                    session.query(Client).filter(Client.id == client_id).first()
                )
                if client_instance:
                    client_instance.client_config = configs
                    session.flush()
                    session.commit()
                    return True
                return False
            except SQLAlchemyError:
                session.rollback()
                raise

    def delete_client(self, client_id: int):
        """Delete a specific client"""
        with get_session() as session:
            try:
                client_instance = (
                    session.query(Client).filter(Client.id == client_id).first()
                )
                if client_instance:
                    session.delete(client_instance)
                    session.flush()
                    session.commit()
                    return True
                return False
            except SQLAlchemyError:
                session.rollback()
                raise
