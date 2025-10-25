from langchain_community.retrievers import BM25Retriever
from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import EnsembleRetriever
from federated_pneumonia_detection.config.settings import Settings
from dotenv import load_dotenv
from federated_pneumonia_detection.src.boundary.CRUD.fetch_documents import (
    fetch_all_documents,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
# TODO: Add axriv MCP server in order to access different research papers, make sure it's functional and really helps.

class QueryEngine:
    def __init__(self, max_history: int = 10):
        """
        Initialize QueryEngine with short-term memory for chat history.

        Args:
            max_history: Maximum number of conversation turns to keep in memory
        """
        self.max_history = max_history
        self.conversation_history: Dict[str, List[Tuple[str, str]]] = {}

        try:
            self.vector_store = PGVector(
                connection=Settings().get_postgres_db_uri(),
                collection_name="research_papers",
                embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            )
        except Exception as e:
            logger.error(f"Error initializing the vectorstore: {e}")
            raise e
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        except Exception as e:
            logger.error(f"Error initializing the llm: {e}")
            raise e
        try:
            self.vector_store_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        except Exception as e:
            logger.error(f"Error initializing the vectorstore retriever: {e}")
            raise e
        try:
            self.documents = fetch_all_documents()
        except Exception as e:
            logger.error(f"Error fetching the documents: {e}")
            raise e
        try:
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 10
        except Exception as e:
            logger.error(f"Error initializing the bm25 retriever: {e}")
            raise e
        try:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_store_retriever],
                llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
                weights=[0.5, 0.5]
            )
        except Exception as e:
            logger.error(f"Error initializing the ensemble retriever: {e}")
            raise e

    def add_to_history(self, session_id: str, user_message: str, ai_response: str):
        """
        Add a conversation turn to the session history.

        Args:
            session_id: Unique identifier for the conversation session
            user_message: The user's query
            ai_response: The AI's response
        """
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        self.conversation_history[session_id].append((user_message, ai_response))

        # Keep only the last max_history turns
        if len(self.conversation_history[session_id]) > self.max_history:
            self.conversation_history[session_id] = self.conversation_history[
                session_id
            ][-self.max_history :]

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Retrieve conversation history for a session.

        Args:
            session_id: Unique identifier for the conversation session

        Returns:
            List of (user_message, ai_response) tuples
        """
        return self.conversation_history.get(session_id, [])

    def clear_history(self, session_id: str):
        """
        Clear conversation history for a session.

        Args:
            session_id: Unique identifier for the conversation session
        """
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]

    def format_history_for_context(self, session_id: str) -> str:
        """
        Format conversation history as context string.

        Args:
            session_id: Unique identifier for the conversation session

        Returns:
            Formatted history string
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        formatted = "Previous conversation:\n"
        for user_msg, ai_msg in history:
            formatted += f"User: {user_msg}\nAssistant: {ai_msg}\n\n"
        return formatted

    def query(self, query: str):
        try:
            results = self.ensemble_retriever.invoke(query)
            return results
        except Exception as e:
            logger.error(f"Error querying the ensemble retriever: {e}")
            return []

    def get_prompts(self, include_history: bool = False):
        try:
            if include_history:
                system_prompt = (
                    "Use the given context and conversation history to answer the question. "
                    "If you don't know the answer, say you don't know. "
                    "Use three sentence maximum and keep the answer concise. "
                    "Previous conversation: {history}\n"
                    "Context: {context}"
                )
            else:
                system_prompt = (
                    "Use the given context to answer the question. "
                    "If you don't know the answer, say you don't know. "
                    "Use three sentence maximum and keep the answer concise. "
                    "Context: {context}"
                )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
        except Exception as e:
            logger.error(f"Error getting the system prompt: {e}")
            raise e
        return prompt

    def get_chain(self, session_id: str = None, include_history: bool = False):
        """
        Get the retrieval chain, optionally with conversation history.

        Args:
            session_id: Session ID for retrieving history
            include_history: Whether to include conversation history in the prompt

        Returns:
            LangChain retrieval chain
        """
        try:
            prompt = self.get_prompts(include_history)
        except Exception as e:
            logger.error(f"Error getting the prompt: {e}")
            raise e
        try:
            chain = create_retrieval_chain(
                self.ensemble_retriever, create_stuff_documents_chain(self.llm, prompt)
            )
        except Exception as e:
            logger.error(f"Error getting the chain: {e}")
            raise e
        return chain

    def query_with_history(self, query: str, session_id: str):
        """
        Query with conversation history context.

        Args:
            query: User query
            session_id: Session ID for conversation tracking

        Returns:
            Dict with answer and context
        """
        try:
            chain = self.get_chain(session_id, include_history=True)
            history_context = self.format_history_for_context(session_id)

            result = chain.invoke({"input": query, "history": history_context})

            # Store this interaction in history
            self.add_to_history(session_id, query, result.get("answer", ""))

            return result
        except Exception as e:
            logger.error(f"Error querying with history: {e}")
            raise e


if __name__ == "__main__":
    query_engine = QueryEngine()
    chain = query_engine.get_chain()
    result = chain.invoke({"input": "What is the point of federated learning?"})
    print(result["answer"])
    result2 = chain.invoke(
        {"input": "What are the main components of federated learning?"}
    )
    print(result2["answer"])
    for result in result["context"]:
        print(result.metadata["source"])
        print("-" * 100)
        print(result.metadata["page"])
