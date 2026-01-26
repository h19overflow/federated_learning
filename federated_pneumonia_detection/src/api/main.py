# >   uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from federated_pneumonia_detection.src.api.endpoints.chat import (
    chat_router,
)
from federated_pneumonia_detection.src.api.endpoints.configuration_settings import (
    configuration_endpoints,
)
from federated_pneumonia_detection.src.api.endpoints.experiments import (
    centralized_endpoints,
    federated_endpoints,
    status_endpoints,
)
from federated_pneumonia_detection.src.api.endpoints.inference.batch_prediction_endpoints import (
    router as inference_batch_router,
)
from federated_pneumonia_detection.src.api.endpoints.inference.gradcam_endpoints import (
    router as inference_gradcam_router,
)
from federated_pneumonia_detection.src.api.endpoints.inference.health_endpoints import (
    router as inference_health_router,
)
from federated_pneumonia_detection.src.api.endpoints.inference.single_prediction_endpoint import (
    router as inference_prediction_router,
)
from federated_pneumonia_detection.src.api.endpoints.reports import (
    report_router,
)
from federated_pneumonia_detection.src.api.endpoints.runs_endpoints import (
    router as runs_endpoints_router,
)
from federated_pneumonia_detection.src.api.middleware import (
    register_exception_handlers,
)
from federated_pneumonia_detection.src.api.middleware.security import (
    MaliciousPromptMiddleware,
)
from federated_pneumonia_detection.src.api.services.startup import (
    initialize_services,
    shutdown_services,
)
from federated_pneumonia_detection.src.internals.loggers.logging_config import (
    configure_logging,
    request_id_ctx,
)
from federated_pneumonia_detection.config.settings import get_settings

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
logger = logging.getLogger(__name__)
logger.info(f"Loaded environment from: {env_path}")

# Get settings (singleton)
settings = get_settings()

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and track unique request IDs for distributed tracing."""

    async def dispatch(self, request, call_next):
        # Generate or use existing request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request state for access in endpoints
        request.state.request_id = request_id

        # Set request ID in context for logging
        token = request_id_ctx.set(request_id)

        try:
            # Process request
            response = await call_next(request)
        finally:
            # Reset context
            request_id_ctx.reset(token)

        # Add to response headers
        response.headers["X-Request-ID"] = request_id

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup/shutdown.

    Delegates to startup service for initialization and cleanup of:
    - Database tables
    - WebSocket metrics server
    - MCP Manager (arxiv integration)
    - Chat services (ArxivEngine, QueryEngine with app.state singletons)
    - W&B inference tracker
    """
    configure_logging()
    await initialize_services(app)
    yield
    await shutdown_services(app)


app = FastAPI(
    title="Federated Pneumonia Detection API",
    description="API for the Federated Pneumonia Detection system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(MaliciousPromptMiddleware)

register_exception_handlers(app)


@app.get("/")
async def read_root():
    return {"message": "Federated Pneumonia Detection API"}


app.include_router(configuration_endpoints.router)
app.include_router(centralized_endpoints.router)
app.include_router(federated_endpoints.router)
app.include_router(status_endpoints.router)
app.include_router(runs_endpoints_router)
app.include_router(chat_router)
app.include_router(inference_health_router)
app.include_router(inference_prediction_router)
app.include_router(inference_batch_router)
app.include_router(inference_gradcam_router)
app.include_router(report_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "federated_pneumonia_detection.src.api.main:app",
        host="127.0.0.1",
        port=8001,
    )
