"""Chat endpoint router assembly."""

from fastapi import APIRouter

from federated_pneumonia_detection.src.api.endpoints.chat.chat_history import (
    router as history_router,
)

from federated_pneumonia_detection.src.api.endpoints.chat.chat_sessions import (
    router as sessions_router,
)
from federated_pneumonia_detection.src.api.endpoints.chat.chat_status import (
    router as status_router,
)
from federated_pneumonia_detection.src.api.endpoints.chat.chat_stream import (
    router as stream_router,
)

router = APIRouter(prefix="/chat", tags=["chat"])
router.include_router(sessions_router)
router.include_router(status_router)
router.include_router(stream_router)
router.include_router(history_router)
