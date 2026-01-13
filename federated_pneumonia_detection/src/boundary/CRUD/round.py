from typing import List, Optional
from datetime import datetime
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.models import Round, Client

class RoundCRUD:


    def create_round(self,client_id: int, round_number: int, round_metadata: Optional[dict] = None):
        """Create a new round for a specific client"""
        if self.get_round_by_client_and_number(client_id, round_number):
            return self.get_round_by_client_and_number(client_id, round_number)
        else:
            with get_session() as session:
                new_round = Round(
                    client_id=client_id,
                    round_number=round_number,
                    round_metadata=round_metadata,
                    start_time=datetime.utcnow(),
                )
                session.add(new_round)
                session.flush()
                session.commit()
                return self.get_round_by_id(new_round.id)

    def get_round_by_id(self,round_id: int) -> Optional[Round]:
        """Get a specific round by ID"""
        with get_session() as session:
            round_instance = session.query(Round).filter(Round.id == round_id).first()
            if round_instance:
                # Expunge to detach from session before closing
                session.expunge(round_instance)
                session.commit()
            return round_instance

    def get_round_by_client_and_number(self,client_id: int, round_number: int) -> Optional[Round]:
        """Get a specific round for a client by round number"""
        with get_session() as session:
            round_instance = session.query(Round).filter(
                Round.client_id == client_id,
                Round.round_number == round_number
            ).first()
            if round_instance:
                session.expunge(round_instance)
            return round_instance

    def get_rounds_by_client_id(self,client_id: int) -> List[Round]:
        """Get all rounds for a specific client"""
        with get_session() as session:
            rounds = session.query(Round).filter(Round.client_id == client_id).order_by(Round.round_number).all()
            for round_instance in rounds:
                session.expunge(round_instance)

            return rounds

    def get_rounds_by_number(self,round_number: int) -> List[Round]:
        """Get all client rounds for a specific round number (global round view)"""
        with get_session() as session:
            rounds = session.query(Round).filter(Round.round_number == round_number).all()
            for round_instance in rounds:
                session.expunge(round_instance)
            return rounds

    def get_all_rounds_by_run(self, run_id: int) -> List[Round]:
        """Get all rounds for a specific run (across all clients)"""
        with get_session() as session:
            # Join with Client table to access run_id
            rounds = session.query(Round).join(Client).filter(
                Client.run_id == run_id
            ).order_by(Round.round_number, Round.client_id).all()
            for round_instance in rounds:
                session.expunge(round_instance)
            return rounds

    def start_round(self,round_id: int):
        """Mark a round as started with current timestamp"""
        with get_session() as session:
            round_instance = session.query(Round).filter(Round.id == round_id).first()
            if round_instance:
                round_instance.start_time = datetime.utcnow()
                session.flush()
                session.commit()
                return True
            return False

    def complete_round(self,round_id: int):
        """Mark a round as completed with current timestamp"""
        with get_session() as session:
            round_instance = session.query(Round).filter(Round.id == round_id).first()
            if round_instance:
                round_instance.end_time = datetime.utcnow()
                session.flush()
                session.commit()
                return True
            return False

    def update_round_metadata(self,round_id: int, metadata: dict):
        """Update round metadata (for storing aggregation info, weights, etc.)"""
        with get_session() as session:
            round_instance = session.query(Round).filter(Round.id == round_id).first()
            if round_instance:
                round_instance.round_metadata = metadata
                session.flush()
                session.commit()
                return True
            return False

    def delete_round(self,round_id: int):
        """Delete a specific round"""
        with get_session() as session:
            round_instance = session.query(Round).filter(Round.id == round_id).first()
            if round_instance:
                session.delete(round_instance)
                session.flush()
                session.commit()
                return True
            return False

    def get_or_create_round(self, db_session, client_id: int, round_number: int) -> int:
        """
        Get existing round or create it if not exists.
        Works with an existing database session.
        
        Args:
            db_session: SQLAlchemy session
            client_id: Client ID to associate round with
            round_number: Round number
            
        Returns:
            round_id: The ID of the round (existing or newly created)
        """
        # Check if round already exists
        existing_round = db_session.query(Round).filter(
            Round.client_id == client_id,
            Round.round_number == round_number
        ).first()
        
        if existing_round:
            return existing_round.id
        
        # Create new round
        new_round = Round(
            client_id=client_id,
            round_number=round_number,
            start_time=datetime.utcnow(),
            round_metadata={'created_at_utc': datetime.utcnow().isoformat()}
        )
        db_session.add(new_round)
        db_session.flush()
        
        return new_round.id


round_crud = RoundCRUD()