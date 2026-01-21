import logging

from langchain_core.documents import Document
from sqlalchemy import MetaData, Table

from federated_pneumonia_detection.src.boundary.engine import get_engine, get_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_all_documents():
    try:
        engine = get_engine()
        session = get_session()
        metadata = MetaData()
        embedding_table = Table(
            "langchain_pg_embedding",
            metadata,
            autoload_with=engine,
        )
        results = session.query(embedding_table).all()
        documents = []
        for row in results:
            doc_content = getattr(row, "document")
            metadata_dict = getattr(row, "cmetadata")
            documents.append(Document(page_content=doc_content, metadata=metadata_dict))
        return documents
    except Exception as e:
        logger.error(f"Error fetching all documents: {e}")
        return []


if __name__ == "__main__":
    documents = fetch_all_documents()
    print(documents)
