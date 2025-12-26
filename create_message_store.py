from federated_pneumonia_detection.src.boundary.engine import settings
from langchain_postgres import PostgresChatMessageHistory
import psycopg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import uuid
...
def create_message_store():
    conn_info = settings.get_postgres_db_uri()
    try:
        with psycopg.connect(conn_info) as conn:
            history = PostgresChatMessageHistory(
                "message_store",
                str(uuid.uuid4()),
                sync_connection=conn
            )
            logger.info("Calling create_tables()...")
            history.create_tables(conn, "message_store")
            logger.info("Successfully called create_tables().")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")

if __name__ == "__main__":
    create_message_store()
