
# >   uvicorn federated_pneumonia_detection.src.api.main:app --reload --host 127.0.0.1 --port 8001
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

from federated_pneumonia_detection.src.api.endpoints.chat import (
    chat_router,
)

# FIXME: ERROR ON logging 
# FIXME: ERROR on notifiying that the training is over.
app = FastAPI(
    title="Federated Pneumonia Detection API",
    description="API for the Federated Pneumonia Detection system",
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
       "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Federated Pneumonia Detection API"}

# FIXME
# TODO , After signling the start of training in cenetralized the websocket disconnects ,
# and when the training is over nothing happens no refersh nothing
app.include_router(configuration_endpoints.router)
app.include_router(centralized_endpoints.router)
app.include_router(federated_endpoints.router)
app.include_router(comparison_endpoints.router)
app.include_router(status_endpoints.router)
app.include_router(logging_endpoints.router)
app.include_router(results_endpoints.router)
app.include_router(chat_router)
