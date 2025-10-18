from federated_pneumonia_detection.src.boundary.vdb_query_engine import QueryEngine
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.schemas import ChatState

def fetch_context(state: ChatState) -> ChatState:
    """Fetch the context from the database."""
    try:
        query_engine = QueryEngine()
        results = query_engine.query(state.query)
        state.ctx = results
        return state
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__": 
   result=  fetch_context(ChatState(query="I have a question about federated learning for pneumonia detection, please answer me"))
   print(result)