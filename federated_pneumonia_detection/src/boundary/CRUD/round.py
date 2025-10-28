from ..engine import  get_session
from ..engine import Round
from typing import List, Optional
from datetime import datetime

class RoundCRUD:
    def create_round(self,client_id: int, round_number: int, round_metadata: Optional[dict] = None):
        """Create a new round for a specific client"""
        with get_session() as session:
            new_round = Round(
                client_id=client_id,
                round_number=round_number,
                round_metadata=round_metadata,
                start_time=datetime.utcnow(),
            )
            session.add(new_round)
            session.flush()  # Get the ID before committing
            round_id = new_round.id

        # Return a fresh instance to avoid detached instance issues
        return RoundCRUD.get_round_by_id(round_id)

    def get_round_by_id(self,round_id: int) -> Optional[Round]:
        """Get a specific round by ID"""
        with get_session() as session:
            round_instance = session.query(Round).filter(Round.id == round_id).first()
            if round_instance:
                # Expunge to detach from session before closing
                session.expunge(round_instance)
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
            from ..engine import Client
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
                return True
            return False

    def complete_round(self,round_id: int):
        """Mark a round as completed with current timestamp"""
        with get_session() as session:
            round_instance = session.query(Round).filter(Round.id == round_id).first()
            if round_instance:
                round_instance.end_time = datetime.utcnow()
                session.flush()
                return True
            return False

    def update_round_metadata(self,round_id: int, metadata: dict):
        """Update round metadata (for storing aggregation info, weights, etc.)"""
        with get_session() as session:
            round_instance = session.query(Round).filter(Round.id == round_id).first()
            if round_instance:
                round_instance.round_metadata = metadata
                session.flush()
                return True
            return False

    def delete_round(self,round_id: int):
        """Delete a specific round"""
        with get_session() as session:
            round_instance = session.query(Round).filter(Round.id == round_id).first()
            if round_instance:
                session.delete(round_instance)
                session.flush()
                return True
            return False