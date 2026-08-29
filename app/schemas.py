from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class PredictionResult(BaseModel):
    id: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    image_url: Optional[str] = None
    gradcam_url: Optional[str] = None
    model_name: str
    timestamp: str

class HistoryItem(BaseModel):
    id: str
    predicted_class: str
    confidence: float
    timestamp: str
    image_url: Optional[str] = None

class ModelInfo(BaseModel):
    model_name: str
    accuracy: float
    parameters: int
    trainable_parameters: int
    inference_fps: float
    model_size_mb: Optional[float] = None

class DatasetInfo(BaseModel):
    total_images: int
    classes: List[str]
    split_ratio: Dict[str, float]

class ResearchInfo(BaseModel):
    dataset: DatasetInfo
    models: List[ModelInfo]
    methodology: List[str]
