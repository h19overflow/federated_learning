from fastapi import WebSocket, APIRouter, WebSocketDisconnect
from federated_pneumonia_detection.src.utils.webocket_logger import ConnectionManager, WebSocketLogHandler

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