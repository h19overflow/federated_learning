import pytest
import uuid


@pytest.fixture
def session_id():
    return str(uuid.uuid4())


def test_create_session(client, mock_session_manager):
    """Test creating a new chat session.

    Asserts:
    - Status code is 200
    - Session ID and title are returned correctly
    - SessionManager.create_session was called
    """
    payload = {"title": "New Session", "initial_query": "Hello"}
    response = client.post("/chat/sessions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-session-id"
    assert data["title"] == "Test Session"
    mock_session_manager.create_session.assert_called_once()


def test_get_history(client, mock_orchestrator, session_id):
    """Test retrieving chat history.

    Asserts:
    - Status code is 200
    - History list is returned with correct content
    - Agent.history was called with session_id
    """
    # Mock agent.history to return a list of tuples
    mock_orchestrator.history.return_value = [("User message", "AI response")]

    response = client.get(f"/chat/history/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert len(data["history"]) == 1
    assert data["history"][0]["user"] == "User message"
    assert data["history"][0]["assistant"] == "AI response"
    mock_orchestrator.history.assert_called_once_with(session_id)


def test_get_session_404(client, mock_session_manager, session_id):
    """Test deleting a non-existent session returns 404.

    Asserts:
    - Status code is 404
    - Error detail is 'Session not found'
    """
    mock_session_manager.delete_session.return_value = False
    response = client.delete(f"/chat/sessions/{session_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_list_sessions(client, mock_session_manager):
    """Test listing all chat sessions.

    Asserts:
    - Status code is 200
    - Returns a list of sessions
    """
    mock_session = pytest.importorskip("unittest.mock").MagicMock()
    mock_session.id = "s1"
    mock_session.title = "Session 1"
    mock_session.created_at.isoformat.return_value = "2023-01-01T00:00:00"
    mock_session.updated_at.isoformat.return_value = "2023-01-01T00:00:00"

    mock_session_manager.list_sessions.return_value = [mock_session]

    response = client.get("/chat/sessions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == "s1"
