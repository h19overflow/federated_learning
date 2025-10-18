from fastapi import FastAPI , HTTPException,middleware
from federated_pneumonia_detection.src.api.settings import Settings


app = FastAPI(title="Federated Pneumonia Detection API",
              version=Settings.API_VERSION,
              description="API for the Federated Pneumonia Detection system",
              docs_url=f"{Settings.API_PREFIX}/docs")


@app.get("/")
async def read_root():
    return {"message": "Federated Pneumonia Detection API"}


