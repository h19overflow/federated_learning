from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from typing import Optional
from langchain_core.documents import Document

class ChatState(BaseModel):
    """State for the chat agent graph."""
    ctx: Optional[list[Document]] = Field(None, description="The context from the database")
    query: Optional[str] = Field(None, description="The query to the database")
    response: Optional[str] = Field(None, description="The response to the query")
    sources: Optional[list[str]] = Field(None, description="The sources of the response")
