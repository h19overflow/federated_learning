from fastapi import FastAPI , HTTPException,middleware



app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Federated Pneumonia Detection API"}


