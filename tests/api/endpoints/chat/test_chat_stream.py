import json
import pytest
import uuid


@pytest.fixture
def session_id():
    return str(uuid.uuid4())


def test_stream_happy_path(client, session_id):
    """Test streaming chat response happy path.

    Asserts:
    - Status code is 200
    - Content-Type is text/event-stream
    - Events follow expected sequence (SESSION, TOKEN, DONE)
    """
    payload = {"query": "Hello", "session_id": session_id, "arxiv_enabled": False}
    response = client.post("/chat/query/stream", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # Parse SSE events
    lines = response.text.strip().split("\n")
    events = []
    for line in lines:
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Assert correct data received
    # We use strings instead of AgentEventType to avoid heavy imports during collection
    assert any(event["type"] == "session" for event in events)
    assert any(
        event["type"] == "token" and event["content"] == "Hello" for event in events
    )
    assert any(
        event["type"] == "token" and event["content"] == " world" for event in events
    )
    assert any(event["type"] == "done" for event in events)


def test_sse_format(client, session_id):
    """Test that the response follows SSE format.

    Asserts:
    - Every non-empty line starts with 'data: '
    - Data is valid JSON
    """
    payload = {"query": "Test SSE format", "session_id": session_id}
    response = client.post("/chat/query/stream", json=payload)
    assert response.status_code == 200

    lines = [line for line in response.text.strip().split("\n") if line]
    assert len(lines) > 0
    for line in lines:
        assert line.startswith("data: ")
        # Ensure it's valid JSON after 'data: '
        data = json.loads(line[6:])
        assert isinstance(data, dict)


def test_stream_validation_error(client):
    """Test streaming with invalid payload (missing required query).

    Asserts:
    - Status code is 422
    """
    # Missing 'query' which is required in ChatMessage
    payload = {"session_id": "test-session"}
    response = client.post("/chat/query/stream", json=payload)
    assert response.status_code == 422
