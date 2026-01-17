# >   uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# CRITICAL: Explicitly load .env file for environment variables
# This ensures database credentials are available for subprocesses
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
from federated_pneumonia_detection.src.api.endpoints.inference.prediction_endpoints import (
    router as inference_prediction_router,
)
from federated_pneumonia_detection.src.api.endpoints.inference.report_endpoints import (
    router as inference_report_router,
)
from federated_pneumonia_detection.src.api.endpoints.runs_endpoints import (
    router as runs_endpoints_router,
)
from federated_pneumonia_detection.src.api.endpoints.streaming import sse_endpoints
from federated_pneumonia_detection.src.api.middleware.security import (
    MaliciousPromptMiddleware,
)
from federated_pneumonia_detection.src.boundary.engine import create_tables
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.mcp_manager import (
    MCPManager,
)
from federated_pneumonia_detection.src.control.dl_model.utils.data.wandb_inference_tracker import (
    get_wandb_tracker,
)
from federated_pneumonia_detection.src.utils.loggers.logging_config import (
    configure_logging,
)

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configure centralized logging (silences third-party libs)
configure_logging()

logger = logging.getLogger(__name__)
logger.info(f"Loaded environment from: {env_path}")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ':16:8'

# Get MCP manager singleton
mcp_manager = MCPManager.get_instance()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup/shutdown.

    Manages WebSocket server and MCP manager lifecycle.
    """
    try:
        logger.info("Ensuring database tables exist...")
        create_tables()
        logger.info("Database tables verified/created")
    except Exception as e:
        # Critical failure: app cannot operate without a database
        logger.critical(f"DATABASE INITIALIZATION FAILED: {e}")
        logger.critical("Cannot proceed with startup. Shutting down.")
        raise

    # Initialize SSE Event Manager
    from federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager import (
        get_sse_event_manager,
    )

    event_manager = get_sse_event_manager()
    logger.info("SSE Event Manager initialized")

    try:
        await mcp_manager.initialize()
        logger.info(
            "MCP manager initialized successfully (arxiv integration available)"
        )
    except ConnectionError as e:
        # Network issue: ArXiv might be temporarily down
        logger.warning(
            f"MCP initialization failed - network issue: {e} "
            "(arxiv search will be unavailable)"
        )
    except ImportError as e:
        # Missing dependency: websockets or other lib not installed
        logger.warning(
            f"MCP initialization failed - missing dependency: {e} "
            "(arxiv search disabled)"
        )
    except Exception as e:
        # Catch-all for unexpected errors in MCP initialization
        logger.warning(
            f"MCP initialization failed (unexpected error): {e} "
            "(arxiv search will be unavailable, but app continues)"
        )

    try:
        tracker = get_wandb_tracker()
        if tracker.initialize(
            entity="projectontheside25-multimedia-university",
            project="FYP2",
            job_type="inference",
        ):
            logger.info(
                "W&B inference tracker initialized (experiment tracking enabled)"
            )
        else:
            logger.warning(
                "W&B tracker rejected configuration (check credentials). "
                "Experiment tracking will be unavailable."
            )
    except ConnectionError as e:
        logger.warning(
            f"W&B connection failed: {e} "
            "(experiment tracking disabled, but training continues)"
        )
    except ImportError as e:
        logger.warning(
            f"W&B not installed: {e} (install with: pip install wandb) "
            "(experiment tracking disabled)"
        )
    except Exception as e:
        logger.warning(
            f"W&B initialization failed (unexpected): {e} "
            "(experiment tracking will be unavailable)"
        )

    yield

    try:
        from federated_pneumonia_detection.src.boundary.engine import dispose_engine

        dispose_engine()
    except Exception as e:
        logger.warning(f"Error disposing database connections: {e}")

    try:
        await mcp_manager.shutdown()
        logger.info("MCP manager shutdown complete")
    except Exception as e:
        logger.warning(
            f"MCP manager shutdown had issues: {e} "
            "(this is non-fatal, app still shutting down)"
        )
    try:
        tracker = get_wandb_tracker()
        tracker.finish()
        logger.info("W&B inference tracker shutdown complete")
    except Exception as e:
        logger.warning(
            f"W&B tracker shutdown had issues: {e} "
            "(this is non-fatal, app still shutting down)"
        )


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
app.include_router(status_endpoints.router)
app.include_router(runs_endpoints_router)
app.include_router(chat_router)
app.include_router(inference_health_router)
app.include_router(inference_prediction_router)
app.include_router(inference_batch_router)
app.include_router(inference_report_router)
app.include_router(sse_endpoints.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "federated_pneumonia_detection.src.api.main:app",
        host="127.0.0.1",
        port=8001,
    )
