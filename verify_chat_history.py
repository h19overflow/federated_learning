import os
import sys
import uuid
from datetime import datetime

# Set PYTHONPATH to root of project
project_root = "c:/Users/User/Projects/FYP2"
sys.path.append(project_root)

# Load environment variables if needed
# os.environ["POSTGRES_DB_URI"] = "..."

try:
    from federated_pneumonia_detection.src.boundary.engine import create_tables, get_session
    from federated_pneumonia_detection.src.boundary.CRUD.chat_history import (
        create_chat_session, get_chat_session, get_all_chat_sessions, delete_chat_session
    )
    from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.arxiv_agent import ArxivAugmentedEngine

    print("--- 1. Verification of Table Creation ---")
    engine = create_tables()
    print("Tables created successfully.")

    print("\n--- 2. Verification of CRUD Operations ---")
    test_id = str(uuid.uuid4())
    session = create_chat_session(title="Test Session", session_id=test_id)
    print(f"Created session: {session}")

    fetched = get_chat_session(test_id)
    print(f"Fetched session: {fetched}")

    all_sessions = get_all_chat_sessions()
    print(f"Total sessions: {len(all_sessions)}")

    print("\n--- 3. Verification of Agent History (Persistent) ---")
    engine = ArxivAugmentedEngine()
    test_query = "What is federated learning?"
    test_response = "Federated learning is a machine learning technique..."
    
    engine.add_to_history(test_id, test_query, test_response)
    print(f"Added message to history for session {test_id}")
    
    history = engine.get_history(test_id)
    print(f"Retrieved history: {history}")
    
    if len(history) > 0 and history[0][0] == test_query:
        print("Agent history verification PASSED.")
    else:
        print("Agent history verification FAILED.")

    print("\n--- 4. Cleanup ---")
    delete_chat_session(test_id)
    print(f"Deleted test session {test_id}")

    print("\nVerification Complete.")

except Exception as e:
    print(f"\nVerification FAILED with error: {e}")
    import traceback
    traceback.print_exc()
