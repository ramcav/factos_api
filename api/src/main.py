from fastapi import FastAPI
from .routes import models

app = FastAPI(
    title="FACTOS API",
    description="A collection of AI models for fake news detection",
    version="1.0.0",
)

app.include_router(models.router, prefix="/api", tags=["models"])

@app.get("/")
async def root():
    return {"message": "Welcome to FACTOS API: A collection of AI models for fake news detection"}