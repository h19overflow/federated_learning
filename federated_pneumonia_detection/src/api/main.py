# >   uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
from federated_pneumonia_detection.src.api.endpoints.inference.health_endpoints import (
    router as inference_health_router,
)
from federated_pneumonia_detection.src.api.endpoints.inference.single_prediction_endpoint import (
    router as inference_prediction_router,
)
from federated_pneumonia_detection.src.api.endpoints.runs_endpoints import (
    router as runs_endpoints_router,
)
from federated_pneumonia_detection.src.api.middleware.security import (
    MaliciousPromptMiddleware,
)
from federated_pneumonia_detection.src.api.services.startup import (
    initialize_services,
    shutdown_services,
)

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
logger = logging.getLogger(__name__)
logger.info(f"Loaded environment from: {env_path}")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ':16:8'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup/shutdown.

    Delegates to startup service for initialization and cleanup of:
    - Database tables
    - WebSocket metrics server
    - MCP Manager (arxiv integration)
    - W&B inference tracker
    """
    await initialize_services()
    yield
    await shutdown_services()


app = FastAPI(
    title="Federated Pneumonia Detection API",
    description="API for the Federated Pneumonia Detection system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",  # Vite default dev port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# Security middleware for prompt injection detection
app.add_middleware(MaliciousPromptMiddleware)


@app.get("/")
async def read_root():
    return {"message": "Federated Pneumonia Detection API"}


app.include_router(configuration_endpoints.router)
app.include_router(centralized_endpoints.router)
app.include_router(federated_endpoints.router)
# app.include_router(comparison_endpoints.router)
app.include_router(status_endpoints.router)
app.include_router(runs_endpoints_router)
app.include_router(chat_router)
app.include_router(inference_health_router)
app.include_router(inference_prediction_router)
app.include_router(inference_batch_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "federated_pneumonia_detection.src.api.main:app",
        host="127.0.0.1",
        port=8001,
    )
