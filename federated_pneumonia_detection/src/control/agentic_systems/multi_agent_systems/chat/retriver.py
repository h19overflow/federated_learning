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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


# TODO , add topk to the retriever
class QueryEngine:
    def __init__(self):
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
            self.vector_store_retriever = self.vector_store.as_retriever()
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
        except Exception as e:
            logger.error(f"Error initializing the bm25 retriever: {e}")
            raise e
        try:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_store_retriever],
                llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
            )
        except Exception as e:
            logger.error(f"Error initializing the ensemble retriever: {e}")
            raise e

    def query(self, query: str):
        try:
            results = self.ensemble_retriever.invoke(query)
            return results
        except Exception as e:
            logger.error(f"Error querying the ensemble retriever: {e}")
            return []

    def get_prompts(self):
        try:
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

    def get_chain(self):
        try:
            prompt = self.get_prompts()
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
