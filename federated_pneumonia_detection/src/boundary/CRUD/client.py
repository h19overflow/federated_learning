from typing import List, Optional
import datetime
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.models import Client


# TODO : If clients are part of multiple rounds they are not being recordeed for the multiple rounds only the first round , in the db
class ClientCRUD:
    def create_client(
        self, run_id: int, client_identifier: str, client_config: Optional[dict] = None
    ):
        """Create a new client"""
        session = get_session()
        try:
            new_client = Client(
                run_id=run_id,
                client_identifier=client_identifier,
                created_at=datetime.datetime.utcnow(),
                client_config=client_config,
            )
            session.add(new_client)
            session.commit()
            client_id = new_client.id
            return self.get_client_by_id(client_id)
        finally:
            session.close()

    def get_client_by_id(self, client_id: int) -> Optional[Client]:
        """Get a specific client by ID"""
        session = get_session()
        try:
            client_instance = (
                session.query(Client).filter(Client.id == client_id).first()
            )
            if client_instance:
                session.expunge(client_instance)
            return client_instance
        finally:
            session.close()

    def get_client_by_identifier(
        self, run_id: int, client_identifier: str
    ) -> Optional[Client]:
        """Get a specific client by run_id and client_identifier"""
        session = get_session()
        try:
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
        finally:
            session.close()

    def get_clients_by_run_id(self, run_id: int) -> List[Client]:
        """Get all clients for a specific run"""
        session = get_session()
        try:
            clients = session.query(Client).filter(Client.run_id == run_id).all()
            for client_instance in clients:
                session.expunge(client_instance)
            return clients
        finally:
            session.close()

    def set_client_config(self, client_id: int, configs: dict):
        """Set client configurations"""
        session = get_session()
        try:
            client_instance = (
                session.query(Client).filter(Client.id == client_id).first()
            )
            if client_instance:
                client_instance.client_config = configs
                session.commit()
                return True
            return False
        finally:
            session.close()

    def delete_client(self, client_id: int):
        """Delete a specific client"""
        session = get_session()
        try:
            client_instance = (
                session.query(Client).filter(Client.id == client_id).first()
            )
            if client_instance:
                session.delete(client_instance)
                session.commit()
                return True
            return False
        finally:
            session.close()
