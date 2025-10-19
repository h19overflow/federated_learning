from fastapi import WebSocket, APIRouter, WebSocketDisconnect
from federated_pneumonia_detection.src.utils.webocket_logger import (
    ConnectionManager,
    WebSocketLogHandler,
)

router = APIRouter()
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    return {"message": "WebSocket connected successfully"}


"""
Frontend: Receiving WebSocket Logs in the Browser
1. Create a WebSocket connection
javascript
// Create a WebSocket connection to your FastAPI backend
const ws = new WebSocket("ws://localhost:8000/ws/logs");
2. Define the onmessage handler
javascript
// This function fires every time the backend sends a log message over the WebSocket
ws.onmessage = function(event) {
    // event.data contains the log message sent from the backend
    console.log("Log from backend:", event.data);
    // You can also update your UI dynamically here, e.g.:
    // document.getElementById("logDisplay").innerText += event.data + "\n";
};
3. Handle connect, error, and close events (optional but recommended)
javascript
ws.onopen = function() {
    console.log("WebSocket connection established.");
};

ws.onerror = function(error) {
    console.error("WebSocket Error:", error);
};

ws.onclose = function(event) {
    console.log("WebSocket closed:", event);
};
Full Example (HTML + JS)
xml
<!DOCTYPE html>
<html>
<head>
    <title>Live Logs</title>
</head>
<body>
    <h2>Backend Logs:</h2>
    <pre id="logDisplay"></pre>
    <script>
        const ws = new WebSocket("ws://localhost:8000/ws/logs");

        ws.onmessage = function(event) {
            document.getElementById("logDisplay").innerText += event.data + "\n";
        };

        ws.onopen = function() {
            console.log("WebSocket connected.");
        };

        ws.onclose = function() {
            console.log("WebSocket closed.");
        };
    </script>
</body>
</html>
Result:

"""
