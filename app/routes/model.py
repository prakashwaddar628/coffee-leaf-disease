import json
from fastapi import APIRouter
from app.config import METRICS_PATH, ACTIVE_MODEL_NAME
from app.schemas import ModelInfo

router = APIRouter()

@router.get("/model", response_model=ModelInfo)
async def get_model_info():
    if not METRICS_PATH.exists():
        # Fallback if metrics are missing
        return ModelInfo(
            model_name=ACTIVE_MODEL_NAME,
            accuracy=0.0,
            parameters=0,
            trainable_parameters=0,
            inference_fps=0.0
        )
        
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    return ModelInfo(
        model_name=metrics.get("model_name", ACTIVE_MODEL_NAME),
        accuracy=metrics.get("test_accuracy", 0.0),
        parameters=metrics.get("total_parameters", 0),
        trainable_parameters=metrics.get("trainable_parameters", 0),
        inference_fps=metrics.get("inference_fps", 0.0),
        model_size_mb=metrics.get("model_size_mb")
    )
