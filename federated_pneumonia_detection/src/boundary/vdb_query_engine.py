import logging
from langchain_postgres import PGVector
from federated_pneumonia_detection.config.settings import Settings
from langchain_huggingface import HuggingFaceEmbeddings


class QueryEngine:
    """QueryEngine for the RAG pipeline."""

    def __init__(self):
        """Initialize the QueryEngine."""
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logging.getLogger(__name__).error(f"Error initializing the embeddings: {e}")
            raise e
        try:
            self.vectorstore = PGVector(
                connection=Settings().get_postgres_db_uri(),
                collection_name="research_papers",
                embeddings=self.embeddings,
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error initializing the vectorstore: {e}"
            )
            raise e

    def query(self, query: str):
        """Query the vectorstore."""
        try:
            results = self.vectorstore.similarity_search(query, k=15)
            return results
        except Exception as e:
            logging.getLogger(__name__).error(f"Error querying the vectorstore: {e}")
            return []
