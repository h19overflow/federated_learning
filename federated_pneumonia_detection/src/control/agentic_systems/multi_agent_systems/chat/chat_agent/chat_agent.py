from pydantic_ai import Agent , RunContext

from chat_schema import ChatDeps, ChatResponse

from chat_prompt import ANSWERING_PROMPT


class ChatAgent:
    def __init__(self):
        self.agent = Agent(
            'gemini-2.0-flash',
            output_type=ChatResponse,
            deps_type=ChatDeps,
        )
        
        