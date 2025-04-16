from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter()

@router.get("/models")
async def get_models():
    # TODO: Implement the logic to return a list of models
    # This should look for all the models in a directory and return the information in a standardized format.
    # The format should be a list of dictionaries, each containing the following keys:
    return {"message": "MODEL LIST HERE"}

@router.get("/models/{model_id}")
async def get_model(model_id: int):
    # TODO: Implement the logic to return information for a specific model
    # This should look for the model in a directory and return the information in a standardized format.
    # This means the parameters the model receives, the parameters it returns, and the format of the data it returns.
    # Everything the user needs to know to use the model.
    return {"message": f"RETURN INFORMATION FOR MODEL {model_id}"}

@router.post("/models/{model_id}/predict")
async def predict(model_id: int, text: str):
    # TODO: Implement the prediction logic here
    # This should look for the model in a directory and use it to return a prediction in a standardized format
    return {"message": f"PREDICT USING MODEL {model_id} WITH THE FOLLOWING TEXT INPUT: {text}"}
