from ..engine import get_session, Client
from typing import List, Optional
import datetime

class ClientCRUD:
    @staticmethod
    def create_client(run_id, client_identifier):
        """Create a new client"""
        with get_session() as session:
            new_client = Client(
                run_id=run_id,
                client_identifier=client_identifier,
                created_at=datetime.datetime.utcnow(),
            )
            session.add(new_client)
            session.flush()  # Get the ID before committing
            client_id = new_client.id

        # Return a fresh instance to avoid detached instance issues
        return ClientCRUD.get_client_by_id(client_id)

    def get_client_by_id(self,client_id: int) -> Optional[Client]:
        """Get a specific client by ID"""
        with get_session() as session:
            client_instance = session.query(Client).filter(Client.id == client_id).first()
            if client_instance:
                # Expunge to detach from session before closing
                session.expunge(client_instance)
            return client_instance
    def set_client_config(self,client_id:int,configs:dict):
        """Set client configurations"""
        with get_session() as session:
            client_instance = session.query(Client).filter(Client.id == client_id).first()
            if client_instance:
                client_instance.client_config = configs
                session.commit()