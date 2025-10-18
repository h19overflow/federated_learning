from langchain_core.documents import Document
from pydantic_ai import Agent , RunContext
from dotenv import load_dotenv

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.chat_agent.chat_schema import ChatDeps, ChatResponse
from langchain_core.documents import Document
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.chat_agent.chat_prompt import ANSWERING_PROMPT
from federated_pneumonia_detection.config.settings import Settings
load_dotenv()
# TODO , After establishing the CRUDS operations and saving configuration add an analysis mode where the agent is provided information about the run 
# as well metrics of currrent run results , perhaps the latest , add / command to fetch context about an experiment , and other commands to help the user to analyze the results.

class ChatAgent:
    """Chat agent class."""
    def __init__(self):
        """Initialize the chat agent."""
        self.agent = None
        
    def get_agent(self):
        """Get the chat agent."""
        if self.agent is None:
            self.agent = Agent(
                Settings().BASE_LLM,
                output_type=ChatResponse,
                deps_type=ChatDeps,
            )
            
            
            @self.agent.system_prompt
            def system_prompt(ctx: RunContext[ChatDeps]):
                """System prompt for the chat agent."""
                return ANSWERING_PROMPT.format(query=ctx.deps.query, ctx="\n".join([doc.page_content for doc in ctx.deps.ctx]))
            return self.agent
        
        return self.agent
    
    async def answer_question(self, question: str,ctx: list[Document]) -> ChatResponse:
        """Answer the question."""
        if self.agent is None:
            self.agent = self.get_agent()
        result = await self.agent.run(
            "Can you answer the following question: {question}",
            deps=ChatDeps(
                query=question,
                ctx=ctx
            ),
        )
        return result.output
