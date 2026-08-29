from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import (
    API_TITLE, 
    API_DESCRIPTION, 
    API_VERSION, 
    API_PREFIX,
    CORS_ORIGINS,
    UPLOADS_DIR,
    GRADCAM_DIR
)
from app.model import coffee_model
from app.routes import predict, history, model, research

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model is already loaded in model.py at startup when imported.
    # We could move the explicit loading here, but it's fine.
    print("Starting up Coffee Leaf Disease Detection API...")
    yield
    print("Shutting down API...")

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories for serving images
app.mount("/api/v1/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/api/v1/gradcam", StaticFiles(directory=str(GRADCAM_DIR)), name="gradcam")

# Include routers
app.include_router(predict.router, prefix=API_PREFIX, tags=["Prediction"])
app.include_router(history.router, prefix=API_PREFIX, tags=["History"])
app.include_router(model.router, prefix=API_PREFIX, tags=["Model"])
app.include_router(research.router, prefix=API_PREFIX, tags=["Research"])

@app.get(f"{API_PREFIX}/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": coffee_model.model is not None,
        "device": str(coffee_model.device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
