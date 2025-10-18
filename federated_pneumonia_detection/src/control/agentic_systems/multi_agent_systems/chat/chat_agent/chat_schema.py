from pydantic import BaseModel, Field
from typing import Optional,List
from langchain_core.documents import Document


class ChatDeps(BaseModel):
    ctx: Optional[List[Document]] = Field(None, description="That is retrieved from the database")
    query: str = Field(..., description="The query to the database")
    
    
class ChatResponse(BaseModel):
    response: str = Field(..., description="The response to the query")
    sources: list[str] = Field(..., description="The sources of the response")








    