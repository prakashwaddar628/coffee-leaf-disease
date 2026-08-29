import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
import aiofiles
from pathlib import Path

from app.config import (
    UPLOADS_DIR, 
    GRADCAM_DIR, 
    MAX_FILE_SIZE_BYTES, 
    SUPPORTED_FORMATS,
    ACTIVE_MODEL_NAME
)
from app.model import coffee_model
from app.schemas import PredictionResult
from app.storage import storage

router = APIRouter()

@router.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload JPG, PNG, or WEBP.")
    
    # Read file bytes
    file_bytes = await file.read()
    
    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        
    # Generate unique ID for this prediction
    pred_id = str(uuid.uuid4())
    
    # Save original image
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    image_filename = f"{pred_id}.{ext}"
    image_path = UPLOADS_DIR / image_filename
    
    async with aiofiles.open(image_path, 'wb') as out_file:
        await out_file.write(file_bytes)
        
    # Run prediction
    try:
        predicted_class, confidence, probabilities, tensor = coffee_model.predict(file_bytes)
        
        # Generate Grad-CAM
        gradcam_filename = f"cam_{pred_id}.jpg"
        gradcam_path = GRADCAM_DIR / gradcam_filename
        coffee_model.generate_gradcam(file_bytes, tensor, str(gradcam_path))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
    # Construct result
    result = PredictionResult(
        id=pred_id,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        image_url=f"/api/v1/uploads/{image_filename}",
        gradcam_url=f"/api/v1/gradcam/{gradcam_filename}",
        model_name=ACTIVE_MODEL_NAME,
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Save to history
    storage.save_prediction(result)
    
    return result

@router.get("/predictions/{prediction_id}", response_model=PredictionResult)
async def get_prediction(prediction_id: str):
    prediction = storage.get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction
