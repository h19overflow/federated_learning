# >   uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from federated_pneumonia_detection.src.api.middleware.security import (
    MaliciousPromptMiddleware,
)
from federated_pneumonia_detection.src.api.endpoints.configuration_settings import (
    configuration_endpoints,
)
from federated_pneumonia_detection.src.api.endpoints.experiments import (
    centralized_endpoints,
    federated_endpoints,
    status_endpoints,
)
from federated_pneumonia_detection.src.api.endpoints.runs_endpoints import (
    router as runs_endpoints_router,
)


from federated_pneumonia_detection.src.api.endpoints.chat import (
    chat_router,
)
from federated_pneumonia_detection.src.api.endpoints.inference.health_endpoints import (
    router as inference_health_router,
)
from federated_pneumonia_detection.src.api.endpoints.inference.prediction_endpoints import (
    router as inference_prediction_router,
)
from federated_pneumonia_detection.src.api.endpoints.inference.batch_prediction_endpoints import (
    router as inference_batch_router,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.mcp_manager import (
    MCPManager,
)
from federated_pneumonia_detection.src.boundary.engine import create_tables
from federated_pneumonia_detection.src.control.dl_model.utils.data.wandb_inference_tracker import (
    get_wandb_tracker,
)
import os
from pathlib import Path

# CRITICAL: Explicitly load .env file for environment variables
# This ensures database credentials are available for subprocesses
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
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
    # Startup
    try:
        logger.info("Ensuring database tables exist...")
        create_tables()
        logger.info("Database tables verified/created")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")

    websocket_thread = threading.Thread(
        target=_start_websocket_server, daemon=True, name="WebSocket-Server-Thread"
    )
    websocket_thread.start()
    logger.info("WebSocket server startup initiated in background thread")

    # Initialize MCP manager for arxiv tools
    try:
        await mcp_manager.initialize()
        logger.info("MCP manager initialized")
    except Exception as e:
        logger.warning(f"MCP manager initialization failed (arxiv unavailable): {e}")

    # Initialize W&B inference tracker
    try:
        tracker = get_wandb_tracker()
        if tracker.initialize(
            entity="projectontheside25-multimedia-university",
            project="FYP2",
            job_type="inference",
        ):
            logger.info("W&B inference tracker initialized")
        else:
            logger.warning("W&B inference tracker initialization failed")
    except Exception as e:
        logger.warning(f"W&B tracker initialization failed (tracking disabled): {e}")

    yield

    # Shutdown
    try:
        await mcp_manager.shutdown()
        logger.info("MCP manager shutdown complete")
    except Exception as e:
        logger.error(f"Error during MCP manager shutdown: {e}")

    # Finish W&B run
    try:
        tracker = get_wandb_tracker()
        tracker.finish()
        logger.info("W&B inference tracker shutdown complete")
    except Exception as e:
        logger.warning(f"Error during W&B tracker shutdown: {e}")


app = FastAPI(
    title="Federated Pneumonia Detection API",
    description="API for the Federated Pneumonia Detection system",
    lifespan=lifespan,
)

# Configure CORS for frontend integration
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


# ============================================================================
# WebSocket Server Auto-Start
# ============================================================================


def _start_websocket_server():
    """
    Start the WebSocket metrics relay server in a background thread.

    This allows metrics from training to stream to the frontend automatically
    without requiring a separate terminal/process.
    """
    try:
        import sys
        from pathlib import Path

        # Add project root to path for imports
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import and run the WebSocket server
        import asyncio
        import websockets
        import json
        from typing import Set

        # Store all connected clients
        connected_clients: Set[websockets.WebSocketServerProtocol] = set()

        async def handler(websocket: websockets.WebSocketServerProtocol) -> None:
            """Handle WebSocket connections and broadcast metrics."""
            connected_clients.add(websocket)
            logger.info(
                f"WebSocket client connected. Total clients: {len(connected_clients)}"
            )

            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        message_type = data.get("type", "unknown")

                        logger.debug(
                            f"Broadcasting {message_type} to {len(connected_clients)} clients"
                        )

                        # Broadcast to all clients except sender
                        if connected_clients:
                            tasks = []
                            for client in connected_clients:
                                if client != websocket:
                                    tasks.append(client.send(message))

                            await asyncio.gather(*tasks, return_exceptions=True)

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON received: {e}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")

            except websockets.exceptions.ConnectionClosed:
                logger.debug("Client disconnected")
            except Exception as e:
                logger.error(f"WebSocket handler error: {e}")
            finally:
                connected_clients.discard(websocket)
                logger.debug(
                    f"WebSocket client removed. Total clients: {len(connected_clients)}"
                )

        async def run_server():
            """Run the WebSocket server."""
            host = "localhost"
            port = 8765

            logger.info(f"Starting WebSocket server on ws://{host}:{port}")

            async with websockets.serve(handler, host, port):
                logger.info("[OK] WebSocket metrics server is running")
                # Keep server running
                await asyncio.Future()

        # Run the server
        asyncio.run(run_server())

    except ImportError as e:
        logger.error(
            f"WebSocket server failed to start: {e}. Install with: pip install websockets"
        )
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")


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
