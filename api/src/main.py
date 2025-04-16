from fastapi import FastAPI
from .routes import models

app = FastAPI()

app.include_router(models.router)

@app.get("/")
async def root():
    return {"message": "Welcome to FACTOS API: A collection of AI models for fake news detection"}
