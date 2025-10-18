from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.chat_agent.chat_agent import ChatAgent
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.schemase import ChatState
from federated_pneumonia_detection.src.utils.logger import get_logger
async def answer(state: ChatState) -> ChatState:
    """Answer the question."""
    try:    
        chat_agent = ChatAgent()
        if state.query is None or state.ctx is None:
            raise ValueError("Query and context are required")
        response = await chat_agent.answer_question(state.query, state.ctx)
        state.response = response.response
        state.sources = response.sources
        return state
    except Exception as e:
        get_logger(__name__).error(f"Error answering question: {e}")
        return {"error": str(e)}