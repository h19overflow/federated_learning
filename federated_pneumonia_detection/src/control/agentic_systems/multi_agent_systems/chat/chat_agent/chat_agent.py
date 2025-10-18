from langchain_core.documents import Document
from pydantic_ai import Agent , RunContext
from dotenv import load_dotenv

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.chat_agent.chat_schema import ChatDeps, ChatResponse
from langchain_core.documents import Document
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.chat_agent.chat_prompt import ANSWERING_PROMPT
load_dotenv()

class ChatAgent:
    """Chat agent class."""
    def __init__(self):
        """Initialize the chat agent."""
        self.agent = None
        
    def get_agent(self):
        """Get the chat agent."""
        if self.agent is None:
            self.agent = Agent(
                'gemini-2.0-flash',
                output_type=ChatResponse,
                deps_type=ChatDeps,
            )
            
            
            @self.agent.system_prompt
            def system_prompt(ctx: RunContext[ChatDeps]):
                """System prompt for the chat agent."""
                return ANSWERING_PROMPT.format(query=ctx.deps.query, ctx=ctx.deps.ctx)
            return self.agent
        
        return self.agent
    
    async def answer_question(self, question: str,ctx: list[Document]) -> ChatResponse:
        """Answer the question."""
        if self.agent is None:
            self.agent = self.get_agent()
        return await self.agent.run(
            "I have a question about federated learning for pneumonia detection, please answer me",
            deps=ChatDeps(
                query=question,
                ctx=ctx
            ),
        )
    
if __name__ == "__main__":
    import asyncio
    chat_agent = ChatAgent()
    response = asyncio.run(chat_agent.answer_question("I have a question about federated learning for pneumonia detection, please answer me", [Document(page_content="I am a document", metadata={"source": "test"})]))
    print(response)