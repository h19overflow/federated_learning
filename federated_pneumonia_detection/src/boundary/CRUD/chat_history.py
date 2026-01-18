from typing import List, Optional
from ..models.chat_session import ChatSession
from ..engine import get_session

def create_chat_session(title: Optional[str] = None, session_id: Optional[str] = None) -> ChatSession:
    """
    Create a new chat session.
    """
    db = get_session()
    try:
        new_session = ChatSession(id=session_id, title=title) if session_id else ChatSession(title=title)
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        return new_session
    finally:
        db.close()

def get_chat_session(session_id: str) -> Optional[ChatSession]:
    """
    Get a chat session by ID.
    """
    db = get_session()
    try:
        return db.query(ChatSession).filter(ChatSession.id == session_id).first()
    finally:
        db.close()

def get_all_chat_sessions() -> List[ChatSession]:
    """
    Get all chat sessions, ordered by updated_at descending.
    """
    db = get_session()
    try:
        return db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    finally:
        db.close()

def update_chat_session_title(session_id: str, title: str) -> Optional[ChatSession]:
    """
    Update the title of a chat session.
    """
    db = get_session()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.title = title
            db.commit()
            db.refresh(session)
        return session
    finally:
        db.close()

def delete_chat_session(session_id: str) -> bool:
    """
    Delete a chat session.
    """
    db = get_session()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
            return True
        return False
    finally:
        db.close()
