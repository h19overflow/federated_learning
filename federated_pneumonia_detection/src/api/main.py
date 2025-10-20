from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from federated_pneumonia_detection.src.api.settings import Settings
from federated_pneumonia_detection.src.api.endpoints.configuration_settings import (
    configuration_endpoints,
)
from federated_pneumonia_detection.src.api.endpoints.experiments import (
    centralized_endpoints,
    federated_endpoints,
    comparison_endpoints,
    status_endpoints,
)
from federated_pneumonia_detection.src.api.endpoints.results import (
    logging_endpoints,
    results_endpoints,
)
from federated_pneumonia_detection.src.api.endpoints.logging import (
    logging_websocket,
)


app = FastAPI(
    title="Federated Pneumonia Detection API",
    version=Settings.API_VERSION,
    description="API for the Federated Pneumonia Detection system",
    docs_url=f"{Settings.API_PREFIX}/docs",
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",  # Vite dev server
        "http://localhost:5173",  # Alternative Vite port
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Federated Pneumonia Detection API"}


# Include routers
app.include_router(configuration_endpoints.router)
app.include_router(centralized_endpoints.router)
app.include_router(federated_endpoints.router)
app.include_router(comparison_endpoints.router)
app.include_router(status_endpoints.router)
app.include_router(logging_endpoints.router)
app.include_router(results_endpoints.router)
app.include_router(logging_websocket.router)
