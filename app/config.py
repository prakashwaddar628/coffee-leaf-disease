"""
Central configuration for the FastAPI backend.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORTS_DIR = DATA_DIR / "reports"

# ── Model Configuration ───────────────────────────────────────────────────────
ACTIVE_MODEL_NAME = "mobilenetv3"
ACTIVE_EXPERIMENT = "experiment_007"
MODEL_PATH = RESULTS_DIR / ACTIVE_MODEL_NAME / ACTIVE_EXPERIMENT / "best_model.pth"
METRICS_PATH = RESULTS_DIR / ACTIVE_MODEL_NAME / ACTIVE_EXPERIMENT / "metrics.json"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / ACTIVE_MODEL_NAME / ACTIVE_EXPERIMENT / "classification_report.json"

# ── Class Names (same order as training) ───────────────────────────────────────
CLASS_NAMES = [
    "coffee___cerscospora",
    "coffee___healthy",
    "coffee___miner",
    "coffee___phoma",
    "coffee___red_spider_mite",
    "coffee___rust",
]

# Human-readable class labels
CLASS_LABELS = {
    "coffee___cerscospora": "Cercospora",
    "coffee___healthy": "Healthy",
    "coffee___miner": "Leaf Miner",
    "coffee___phoma": "Phoma",
    "coffee___red_spider_mite": "Red Spider Mite",
    "coffee___rust": "Rust",
}

# ── Image Configuration ───────────────────────────────────────────────────────
IMAGE_SIZE = 224
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ── Preprocessing (same as training) ──────────────────────────────────────────
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# ── Storage ───────────────────────────────────────────────────────────────────
UPLOADS_DIR = PROJECT_ROOT / "app" / "uploads"
GRADCAM_DIR = PROJECT_ROOT / "app" / "gradcam"
PREDICTIONS_FILE = PROJECT_ROOT / "app" / "predictions.json"

# Create directories
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
]

# ── API ───────────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"
API_TITLE = "Coffee Leaf Disease Detection API"
API_DESCRIPTION = "AI-powered coffee leaf disease classification using PyTorch"
API_VERSION = "1.0.0"
