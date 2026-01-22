import uuid

from sqlalchemy import Column, DateTime, String
from sqlalchemy.sql import func

from .base import Base


class ChatSession(Base):
    """
    Model for storing chat sessions and their metadata.
    Actual messages are stored via langchain_postgres.
    """

    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now(),
        server_default=func.now(),
    )

    def __repr__(self):
        return f"<ChatSession(id='{self.id}', title='{self.title}')>"
