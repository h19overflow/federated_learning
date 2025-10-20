from fastapi import FastAPI
from federated_pneumonia_detection.src.api.settings import Settings
from federated_pneumonia_detection.src.api.endpoints.configuration_settings import (
    configuration_endpoints,
)
from federated_pneumonia_detection.src.api.endpoints.experiments import (
    centralized_endpoints,
    federated_endpoints,
    comparison_endpoints,
)
from federated_pneumonia_detection.src.api.endpoints.results import (
    logging_endpoints,
    results_endpoints,
)


app = FastAPI(
    title="Federated Pneumonia Detection API",
    version=Settings.API_VERSION,
    description="API for the Federated Pneumonia Detection system",
    docs_url=f"{Settings.API_PREFIX}/docs",
)


@app.get("/")
async def read_root():
    return {"message": "Federated Pneumonia Detection API"}


# Include routers
app.include_router(configuration_endpoints.router)
app.include_router(centralized_endpoints.router)
app.include_router(federated_endpoints.router)
app.include_router(comparison_endpoints.router)
app.include_router(logging_endpoints.router)
app.include_router(results_endpoints.router)
