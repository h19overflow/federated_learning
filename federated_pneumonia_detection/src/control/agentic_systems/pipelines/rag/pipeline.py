import os
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

from federated_pneumonia_detection.config.settings import Settings
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger


def load_pdf(file_path: str) -> list[Document]:
    try:
        loader = PyPDFium2Loader(file_path)
        return loader.load()
    except Exception as e:
        get_logger(__name__).error(f"Error loading PDF: {e}")
        return []


def chunk_pdf(pdf_docs: list[Document]) -> list[Document]:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chunker = SemanticChunker(embeddings=embeddings)
        return chunker.split_documents(pdf_docs)
    except Exception as e:
        get_logger(__name__).error(f"Error chunking PDF: {e}")
        return []


def insert_documents_to_postgres(postgres_url: str, documents: list[Document]):
    """
    Insert documents into Postgres database.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = PGVector(
            connection=postgres_url,
            collection_name="research_papers",
            embeddings=embeddings,
        )
        vectorstore.add_documents(documents=documents)
        return True
    except Exception as e:
        get_logger(__name__).error(f"Error inserting documents into Postgres: {e}")
        return False


def pipeline(postgres_url: str, file_path: str):
    """
    Pipeline to load PDF, chunk it, and insert it into Postgres database.
    """
    try:
        pdf_docs = load_pdf(file_path)
        chunked_docs = chunk_pdf(pdf_docs)
        insert_documents_to_postgres(postgres_url, chunked_docs)
        return True
    except Exception as e:
        get_logger(__name__).error(f"Error in pipeline: {e}")
        return False


if __name__ == "__main__":
    directory_path = os.path.join(os.path.dirname(__file__), "input")

    for file in os.listdir(directory_path):
        get_logger(__name__).info(f"Processing file: {file}")
        if file.endswith(".pdf"):
            pipeline(
                Settings().get_postgres_db_uri(), os.path.join(directory_path, file)
            )
