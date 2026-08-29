from fastapi import APIRouter
import json
from app.schemas import ResearchInfo, DatasetInfo, ModelInfo
from app.config import PROJECT_ROOT

router = APIRouter()

@router.get("/research", response_model=ResearchInfo)
async def get_research_info():
    # Hardcoded or parsed from dataset stats
    # For now, we will return the known stats of the harmonized dataset
    dataset_info = DatasetInfo(
        total_images=4560,  # 3648 + 456 + 456
        classes=[
            "Cercospora",
            "Healthy",
            "Leaf Miner",
            "Phoma",
            "Red Spider Mite",
            "Rust"
        ],
        split_ratio={
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1
        }
    )
    
    # We fetch current best MobileNet metrics
    mobilenet_metrics_path = PROJECT_ROOT / "results" / "mobilenetv3" / "experiment_007" / "metrics.json"
    mobilenet_info = None
    if mobilenet_metrics_path.exists():
        with open(mobilenet_metrics_path, "r") as f:
            m_data = json.load(f)
            mobilenet_info = ModelInfo(
                model_name="MobileNetV3",
                accuracy=m_data.get("test_accuracy", 0),
                parameters=m_data.get("total_parameters", 0),
                trainable_parameters=m_data.get("trainable_parameters", 0),
                inference_fps=m_data.get("inference_fps", 0),
                model_size_mb=m_data.get("model_size_mb", 6.2)
            )
            
    # And ResNet50 metrics (experiment_004 had partial results on old dataset, 
    # but we will just provide what we have)
    resnet_metrics_path = PROJECT_ROOT / "results" / "resnet50" / "experiment_004" / "metrics.json"
    resnet_info = None
    if resnet_metrics_path.exists():
        with open(resnet_metrics_path, "r") as f:
            r_data = json.load(f)
            resnet_info = ModelInfo(
                model_name="ResNet50",
                accuracy=r_data.get("test_accuracy", 0),
                parameters=r_data.get("total_parameters", 0),
                trainable_parameters=r_data.get("trainable_parameters", 0),
                inference_fps=r_data.get("inference_fps", 0),
                model_size_mb=r_data.get("model_size_mb", 89.7)
            )

    models = []
    if mobilenet_info:
        models.append(mobilenet_info)
    if resnet_info:
        models.append(resnet_info)

    methodology = [
        "Dataset Harmonization (RoCoLe + JMuBEN)",
        "Exploratory Data Analysis (EDA)",
        "Image Preprocessing (Resize, Normalize)",
        "Data Augmentation (Albumentations)",
        "Baseline Model Training (MobileNetV3)",
        "Advanced Model Evaluation (ResNet50)",
        "Explainability (Grad-CAM)"
    ]
    
    return ResearchInfo(
        dataset=dataset_info,
        models=models,
        methodology=methodology
    )
