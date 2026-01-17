from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from federated_pneumonia_detection.config.settings import Settings
from federated_pneumonia_detection.src.boundary.CRUD.fetch_documents import (
    fetch_all_documents,
)

load_dotenv()


class QueryEngine:
    def __init__(self, max_history: int = 10):
        """
        Initialize QueryEngine with short-term memory for chat history.

        Args:
            max_history: Maximum number of conversation turns to keep in memory
        """
        from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent.history import (
            ChatHistoryManager,
        )

        self.max_history = max_history
        self.history_manager = ChatHistoryManager(
            table_name="message_store", max_history=max_history
        )
        # TODO: Reindex the whole embeddings vectorDB to work with gemini embeddings models the current setup has low response time
        # self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-3-flash-preview")
        self.vector_store = PGVector(
            connection=Settings().get_postgres_db_uri(),
            collection_name="research_papers",
            embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
        self.vector_store_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10}
        )
        self.documents = fetch_all_documents()
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 10
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_store_retriever],
            llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview"),
            weights=[0.5, 0.5],
        )

    def add_to_history(self, session_id: str, user_message: str, ai_response: str):
        """
        Add a conversation turn to the session history.

        Args:
            session_id: Unique identifier for the conversation session
            user_message: The user's query
            ai_response: The AI's response
        """
        self.history_manager.add_to_history(session_id, user_message, ai_response)

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Retrieve conversation history for a session.

        Args:
            session_id: Unique identifier for the conversation session

        Returns:
            List of (user_message, ai_response) tuples
        """
        return self.history_manager.get_history(session_id)

    def clear_history(self, session_id: str):
        """
        Clear conversation history for a session.

        Args:
            session_id: Unique identifier for the conversation session
        """
        self.history_manager.clear_history(session_id)

    def format_history_for_context(self, session_id: str) -> str:
        """
        Format conversation history as context string.

        Args:
            session_id: Unique identifier for the conversation session

        Returns:
            Formatted history string
        """
        return self.history_manager.format_for_context(session_id)

    def query(self, query: str):
        try:
            return self.ensemble_retriever.invoke(query)
        except Exception:
            return []

    def get_prompts(self, include_history: bool = False):
        markdown_instructions = (
            "Format your response using Markdown for better readability:\n"
            "- Use **bold** for key terms and important metrics\n"
            "- Use bullet points or numbered lists for multiple items\n"
            "- Use `code` formatting for technical values, percentages, or numbers\n"
            "- Use ### headings to organize longer responses\n"
            "- Use tables when comparing multiple metrics or values\n"
            "- Keep responses well-structured and scannable\n\n"
        )

        if include_history:
            system_prompt = (
                "You are a helpful AI assistant specializing in federated learning and medical imaging. "
                "Use the given context and conversation history to answer the question accurately.\n\n"
                f"{markdown_instructions}"
                "If you don't know the answer, clearly state that you don't have enough information.\n"
                "Provide detailed, informative responses while keeping them well-organized.\n\n"
                "Previous conversation:\n{history}\n\n"
                "Context:\n{context}"
            )
        else:
            system_prompt = (
                "You are a helpful AI assistant specializing in federated learning and medical imaging. "
                "Use the given context to answer the question accurately.\n\n"
                f"{markdown_instructions}"
                "If you don't know the answer, clearly state that you don't have enough information.\n"
                "Provide detailed, informative responses while keeping them well-organized.\n\n"
                "Context:\n{context}"
            )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

    def get_chain(self, session_id: str = None, include_history: bool = False):
        """
        Get the retrieval chain, optionally with conversation history.

        Args:
            session_id: Session ID for retrieving history
            include_history: Whether to include conversation history in the prompt

        Returns:
            LangChain retrieval chain
        """
        prompt = self.get_prompts(include_history)
        return create_retrieval_chain(
            self.ensemble_retriever, create_stuff_documents_chain(self.llm, prompt)
        )

    def query_with_history(
        self, query: str, session_id: str, original_query: Optional[str] = None
    ):
        """
        Query with conversation history context.

        Args:
            query: User query
            session_id: Session ID for conversation tracking

        Returns:
            Dict with answer and context
        """
        chain = self.get_chain(session_id, include_history=True)
        history_context = self.format_history_for_context(session_id)

        result = chain.invoke({"input": query, "history": history_context})

        # Store this interaction in history
        history_query = original_query if original_query is not None else query
        self.add_to_history(session_id, history_query, result.get("answer", ""))

        return result

    async def query_with_history_stream(
        self, query: str, session_id: str, original_query: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream query results token by token with conversation history context.

        Uses synchronous retrieval followed by async LLM streaming to avoid
        async database engine issues with the ensemble retriever.

        Args:
            query: User query
            session_id: Session ID for conversation tracking

        Yields:
            Dict with type and content for each streamed chunk
        """
        try:
            # Retrieve documents synchronously (BM25 + PGVector ensemble)
            retrieved_docs = self.ensemble_retriever.invoke(query)

            # Format retrieved documents as context
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # Get conversation history
            history_context = self.format_history_for_context(session_id)

            # Build the prompt with context and history
            prompt = self.get_prompts(include_history=bool(history_context))
            if history_context:
                messages = prompt.format_messages(
                    context=context, history=history_context, input=query
                )
            else:
                messages = prompt.format_messages(context=context, input=query)

            full_response = ""

            # Stream only the LLM response
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    # Handle content that may be a list (Gemini) or string
                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            part if isinstance(part, str) else part.get("text", "")
                            for part in content
                        )

                    if content:
                        full_response += content
                        yield {"type": "token", "content": content}

            history_query = original_query if original_query is not None else query
            self.add_to_history(session_id, history_query, full_response)
            yield {"type": "done", "session_id": session_id}

        except Exception as e:
            yield {"type": "error", "message": str(e)}
