import logging
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

logger = logging.getLogger(__name__)
load_dotenv()


class QueryEngine:
    def __init__(self, max_history: int = 10):
        """
        Initialize QueryEngine with short-term memory for chat history.

        Args:
            max_history: Maximum number of conversation turns to keep in memory
        """
        from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.history.postgres_history import (  # noqa: E501
            ChatHistoryManager,
        )

        logger.info(f"[QueryEngine] Initializing with max_history={max_history}")
        self.max_history = max_history
        self.history_manager = ChatHistoryManager(
            table_name="message_store",
            max_history=max_history,
        )

        try:
            logger.info("[QueryEngine] Connecting to PGVector store...")
            self.vector_store = PGVector(
                connection=Settings().get_postgres_db_uri(),
                collection_name="research_papers",
                embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            )
            logger.info("[QueryEngine] Vector store connected successfully")
        except Exception as e:
            logger.error(
                f"[QueryEngine] Error initializing the vectorstore: {e}",
                exc_info=True,
            )
            raise e
        try:
            logger.info(
                "[QueryEngine] Initializing ChatGoogleGenerativeAI (gemini-3-flash-preview)...",  # noqa: E501
            )
            self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
            logger.info("[QueryEngine] LLM initialized successfully")
        except Exception as e:
            logger.error(
                f"[QueryEngine] Error initializing the llm: {e}",
                exc_info=True,
            )
            raise e
        try:
            self.vector_store_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 10},
            )
        except Exception as e:
            logger.error(
                f"[QueryEngine] Error initializing the vectorstore retriever: {e}",
                exc_info=True,
            )
            raise e
        try:
            logger.info("[QueryEngine] Fetching documents for BM25 retriever...")
            self.documents = fetch_all_documents()
            logger.info(f"[QueryEngine] Fetched {len(self.documents)} documents")
        except Exception as e:
            logger.error(
                f"[QueryEngine] Error fetching the documents: {e}",
                exc_info=True,
            )
            raise e
        try:
            logger.info("[QueryEngine] Initializing BM25 retriever...")
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 10
            logger.info("[QueryEngine] BM25 retriever initialized")
        except Exception as e:
            logger.error(
                f"[QueryEngine] Error initializing the bm25 retriever: {e}",
                exc_info=True,
            )
            raise e
        try:
            logger.info("[QueryEngine] Initializing EnsembleRetriever...")
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_store_retriever],
                llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview"),
                weights=[0.5, 0.5],
            )
            logger.info("[QueryEngine] EnsembleRetriever initialized successfully")
        except Exception as e:
            logger.error(
                f"[QueryEngine] Error initializing the ensemble retriever: {e}",
                exc_info=True,
            )
            raise e

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
            results = self.ensemble_retriever.invoke(query)
            return results
        except Exception as e:
            logger.error(f"Error querying the ensemble retriever: {e}")
            return []

    def get_prompts(self, include_history: bool = False):
        try:
            markdown_instructions = (
                "Format your response using Markdown for better readability:\n"
                "- Use **bold** for key terms and important metrics\n"
                "- Use bullet points or numbered lists for multiple items\n"
                "- Use `code` formatting for technical values, percentages, or numbers\n"  # noqa: E501
                "- Use ### headings to organize longer responses\n"
                "- Use tables when comparing multiple metrics or values\n"
                "- Keep responses well-structured and scannable\n\n"
            )

            if include_history:
                system_prompt = (
                    "You are a helpful AI assistant specializing in federated learning and medical imaging. "  # noqa: E501
                    "Use the given context and conversation history to answer the question accurately.\n\n"  # noqa: E501
                    f"{markdown_instructions}"
                    "If you don't know the answer, clearly state that you don't have enough information.\n"  # noqa: E501
                    "Provide detailed, informative responses while keeping them well-organized.\n\n"  # noqa: E501
                    "Previous conversation:\n{history}\n\n"
                    "Context:\n{context}"
                )
            else:
                system_prompt = (
                    "You are a helpful AI assistant specializing in federated learning and medical imaging. "  # noqa: E501
                    "Use the given context to answer the question accurately.\n\n"
                    f"{markdown_instructions}"
                    "If you don't know the answer, clearly state that you don't have enough information.\n"  # noqa: E501
                    "Provide detailed, informative responses while keeping them well-organized.\n\n"  # noqa: E501
                    "Context:\n{context}"
                )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ],
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
                self.ensemble_retriever,
                create_stuff_documents_chain(self.llm, prompt),
            )
        except Exception as e:
            logger.error(f"Error getting the chain: {e}")
            raise e
        return chain

    def query_with_history(
        self,
        query: str,
        session_id: str,
        original_query: Optional[str] = None,
    ):
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
            # Store this interaction in history
            history_query = original_query if original_query is not None else query
            self.add_to_history(session_id, history_query, result.get("answer", ""))

            return result
        except Exception as e:
            logger.error(f"Error querying with history: {e}")
            raise e

    async def query_with_history_stream(
        self,
        query: str,
        session_id: str,
        original_query: Optional[str] = None,
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
            logger.info(
                f"[QueryEngine] Starting stream for session {session_id}, query: '{query[:50]}...'",  # noqa: E501
            )

            # Retrieve documents synchronously (BM25 + PGVector ensemble)
            logger.info("[QueryEngine] Invoking ensemble retriever...")
            retrieved_docs = self.ensemble_retriever.invoke(query)
            logger.info(f"[QueryEngine] Retrieved {len(retrieved_docs)} documents")

            # Format retrieved documents as context
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # Get conversation history
            history_context = self.format_history_for_context(session_id)
            if history_context:
                logger.info(
                    f"[QueryEngine] Found conversation history for session {session_id}",  # noqa: E501
                )
            else:
                logger.info(f"[QueryEngine] No history found for session {session_id}")

            # Build the prompt with context and history
            prompt = self.get_prompts(include_history=bool(history_context))

            # Create messages for the LLM
            logger.info("[QueryEngine] Formatting messages for LLM")
            if history_context:
                messages = prompt.format_messages(
                    context=context,
                    history=history_context,
                    input=query,
                )
            else:
                messages = prompt.format_messages(context=context, input=query)

            full_response = ""
            chunk_count = 0

            # Stream only the LLM response
            logger.info("[QueryEngine] Starting LLM streaming...")
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
                        chunk_count += 1
                        full_response += content
                        yield {"type": "token", "content": content}

            logger.info(
                f"[QueryEngine] Streaming completed. Generated {chunk_count} chunks. Response length: {len(full_response)}",  # noqa: E501
            )

            history_query = original_query if original_query is not None else query
            self.add_to_history(session_id, history_query, full_response)
            yield {"type": "done", "session_id": session_id}

        except Exception as e:
            logger.error(
                f"[QueryEngine] Error streaming query with history: {e}",
                exc_info=True,
            )
            yield {"type": "error", "message": str(e)}


if __name__ == "__main__":
    query_engine = QueryEngine()
    chain = query_engine.get_chain()
    result = chain.invoke({"input": "What is the point of federated learning?"})
    print(result["answer"])
    result2 = chain.invoke(
        {"input": "What are the main components of federated learning?"},
    )
    print(result2["answer"])
    for result in result["context"]:
        print(result.metadata["source"])
        print("-" * 100)
        print(result.metadata["page"])
