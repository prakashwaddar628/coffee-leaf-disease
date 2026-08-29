import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from app.config import (
    MODEL_PATH, 
    CLASS_NAMES, 
    IMAGE_SIZE, 
    NORMALIZE_MEAN, 
    NORMALIZE_STD
)

class CoffeeLeafModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
        # Variables for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Hook into the last convolutional layer (for MobileNetV3)
        # In MobileNetV3, the features are extracted by model.features
        # We hook into the last layer of features
        if hasattr(self.model, 'features'):
            target_layer = self.model.features[-1]
            target_layer.register_forward_hook(self.save_activation)
            target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _load_model(self):
        # Initialize MobileNetV3 with the same architecture
        model = models.mobilenet_v3_small(weights=None)
        
        # Replace classifier
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, len(CLASS_NAMES))
        
        # Load weights
        if MODEL_PATH.exists():
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded model from {MODEL_PATH}")
        else:
            print(f"Warning: Model weights not found at {MODEL_PATH}")
            
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2(),
        ])

    def predict(self, image_bytes: bytes):
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        
        transformed = self.transform(image=image_np)
        tensor = transformed["image"].unsqueeze(0).to(self.device)
        
        # Inference
        self.model.zero_grad()
        with torch.enable_grad():
            tensor.requires_grad_(True)
            output = self.model(tensor)
            
            probabilities = F.softmax(output, dim=1).squeeze().detach().cpu().numpy()
            predicted_idx = np.argmax(probabilities)
            
            # Trigger backward pass for Grad-CAM on the predicted class
            score = output[0, predicted_idx]
            score.backward()
            
        # Format results
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        probs_dict = {
            CLASS_NAMES[i]: float(probabilities[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        return predicted_class, confidence, probs_dict, tensor

    def generate_gradcam(self, original_image_bytes: bytes, tensor: torch.Tensor, output_path: str):
        if self.gradients is None or self.activations is None:
            # If hooks didn't capture (e.g. inference without grad), just return original
            image = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
            image.save(output_path)
            return
            
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # ReLU on CAM
        cam = np.maximum(cam, 0)
        
        # Normalize CAM
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        # Resize CAM to image size
        image = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_np = np.array(image)
        
        cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on image
        alpha = 0.5
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
        
        # Save result
        Image.fromarray(overlay).save(output_path)

coffee_model = CoffeeLeafModel()
