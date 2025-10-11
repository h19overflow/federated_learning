from pydantic import BaseModel, Field

class ChatDeps(BaseModel):
    ctx: str = Field(..., description="That is retrieved from the database")
    query: str = Field(..., description="The query to the database")
    
    
class ChatResponse(BaseModel):
    response: str = Field(..., description="The response to the query")
    

    
    