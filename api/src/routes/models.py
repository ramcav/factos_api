from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ..services.model_manager import model_manager

router = APIRouter()

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    model_id: int
    model_name: str

@router.get("/models", response_model=List[Dict[str, Any]])
async def get_models():
    """Return a list of all available fake news detection models."""
    return model_manager.get_models()

@router.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model(model_id: str):
    """Return information about a specific model."""
    model_info = model_manager.get_model_info(model_id)
    if not model_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found"
        )
    return model_info

@router.post("/models/{model_id}/predict", response_model=Dict[str, Any])
async def predict(model_id: str, input_data: TextInput):
    """Use a specific model to predict if the provided text is fake or real news."""
    result = model_manager.predict(model_id, input_data.text)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
    
    return result