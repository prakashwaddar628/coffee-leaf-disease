import nbformat as nbf
from pathlib import Path

def create_notebook():
    nb = nbf.v4.new_notebook()

    # 1. Introduction
    nb.cells.append(nbf.v4.new_markdown_cell(
'''# 1. Introduction
## Universal Training Framework & MobileNetV3 Baseline

**Goal**: Develop a universal training loop capable of evaluating any classification model, starting with MobileNetV3 Small as the baseline transfer-learning architecture.

**Research Questions**:
* **RQ14**: How well does MobileNetV3 perform on Coffee Leaf Disease?
* **RQ15**: Does transfer learning work? (Freezing backbone)
* **RQ16**: Does Early Stopping improve performance?
* **RQ17**: Is AdamW a better optimizer?
* **RQ18**: Can MobileNetV3 be deployed?
'''))

    # 2. Load Configuration & Experiment Setup
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Load Configuration & Experiment Setup"))
    nb.cells.append(nbf.v4.new_code_cell(
'''import yaml
import json
import os
import shutil
from pathlib import Path

# Load universal config
with open('../datasets/config.yaml') as f:
    config = yaml.safe_load(f)['training']

MODEL_NAME = "mobilenetv3"
RESULTS_DIR = Path(f"../results/{MODEL_NAME}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Automatically determine next experiment number
existing_exp = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("experiment_")]
exp_num = len(existing_exp) + 1
EXP_DIR = RESULTS_DIR / f"experiment_{exp_num:03d}"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Save a snapshot of the config for reproducibility
with open(EXP_DIR / "config_snapshot.yaml", 'w') as f:
    yaml.dump(config, f)

print(f"Created Experiment: {EXP_DIR}")
print(f"Hyperparameters: {config}")
'''))

    # 3. Load Dataset
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Load Dataset"))
    nb.cells.append(nbf.v4.new_code_cell(
'''import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import pandas as pd

data_dir = Path('../data')
classes = sorted([d.name for d in (data_dir / 'train').iterdir() if d.is_dir()])

# Same transforms as Sprint 4
IMG_SIZE = config['image_size']
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class CoffeeLeafDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            for img_path in (self.root_dir / cls).glob('*.jpg'):
                self.samples.append((str(img_path), self.class_to_idx[cls]))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label, img_path

train_dataset = CoffeeLeafDataset(data_dir / 'train', transform=train_transform)
val_dataset = CoffeeLeafDataset(data_dir / 'validation', transform=val_transform)
test_dataset = CoffeeLeafDataset(data_dir / 'test', transform=val_transform)

# Weighted sampler for imbalance
class_weights_df = pd.read_csv('../data/reports/class_weights.csv')
class_weights_dict = dict(zip(class_weights_df['Class'], class_weights_df['Weight']))
sample_weights = [class_weights_dict[classes[label]] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=config['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
print(f"Loaded DataLoaders. Train: {len(train_loader)} batches.")
'''))

    # 4 & 5. Device & Reproducibility
    nb.cells.append(nbf.v4.new_markdown_cell("## 4 & 5. Device Auto-Detection & Reproducibility"))
    nb.cells.append(nbf.v4.new_code_cell(
'''import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config['random_seed'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
'''))

    # 6 & 7. Load Model & Transfer Learning
    nb.cells.append(nbf.v4.new_markdown_cell("## 6 & 7. Load Model & Transfer Learning (MobileNetV3 Small)"))
    nb.cells.append(nbf.v4.new_code_cell(
'''import torchvision.models as models
import torch.nn as nn

# Load pretrained MobileNetV3 Small
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# Freeze feature extractor (Backbone)
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, len(classes))

model = model.to(device)
print("MobileNetV3 Small initialized. Backbone Frozen. Classifier Replaced.")
'''))

    # 8, 9, 10, 11. Loss, Optimizer, Scheduler, Early Stopping
    nb.cells.append(nbf.v4.new_markdown_cell("## 8, 9, 10, 11. Training Components Setup"))
    nb.cells.append(nbf.v4.new_code_cell(
'''import torch.optim as optim

# Loss Function with weights (optional since we use WeightedRandomSampler, but good for extra penalty)
criterion = nn.CrossEntropyLoss()

# Optimizer (AdamW)
optimizer = optim.AdamW(model.classifier.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Early Stopping tracker
patience = config['early_stopping']
'''))

    # 12. Universal Training Loop
    nb.cells.append(nbf.v4.new_markdown_cell("## 12. Universal Training Loop"))
    nb.cells.append(nbf.v4.new_code_cell(
'''from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import time

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total
'''))

    # 13 & 14. Training Execution & Best Model Selection
    nb.cells.append(nbf.v4.new_markdown_cell("## 13 & 14. Training Execution"))
    nb.cells.append(nbf.v4.new_code_cell(
'''best_val_loss = float('inf')
epochs_no_improve = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

start_time = time.time()

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    TimeRemainingColumn()
) as progress:
    
    task = progress.add_task("[cyan]Training...", total=config['epochs'])
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Best Model Selection based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), EXP_DIR / "best_model.pth")
        else:
            epochs_no_improve += 1
            
        progress.update(task, advance=1, description=f"[cyan]Epoch {epoch+1}/{config['epochs']} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if epochs_no_improve >= patience:
            print(f"\\nEarly stopping triggered at epoch {epoch+1}!")
            break

torch.save(model.state_dict(), EXP_DIR / "last_model.pth")
train_time = time.time() - start_time
print(f"\\nTraining completed in {train_time:.2f}s")
'''))

    # 15. Testing & Evaluation
    nb.cells.append(nbf.v4.new_markdown_cell("## 15. Testing & Evaluation"))
    nb.cells.append(nbf.v4.new_code_cell(
'''from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import torch.nn.functional as F

# Load best model for testing
model.load_state_dict(torch.load(EXP_DIR / "best_model.pth"))
model.eval()

all_preds = []
all_labels = []
all_probs = []
misclassified = []

start_time = time.time()
with torch.no_grad():
    for images, labels, paths in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        # Track misclassified
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified.append({
                    'path': paths[i],
                    'true_label': classes[labels[i]],
                    'pred_label': classes[preds[i]],
                    'confidence': probs[i][preds[i]].item()
                })
                
inference_time = time.time() - start_time
fps = len(test_dataset) / inference_time

# Metrics
report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
print(classification_report(all_labels, all_preds, target_names=classes))
'''))

    # 16. Visualizations
    nb.cells.append(nbf.v4.new_markdown_cell("## 16. Visualizations"))
    nb.cells.append(nbf.v4.new_code_cell(
'''import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Plot Loss & Acc
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax2.plot(history['train_acc'], label='Train Acc')
ax2.plot(history['val_acc'], label='Val Acc')
ax2.set_title('Accuracy Curve')
ax2.legend()
plt.savefig(EXP_DIR / 'training_curves.png')
plt.show()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(EXP_DIR / 'confusion_matrix.png')
plt.show()
'''))

    # 17. Error Analysis
    nb.cells.append(nbf.v4.new_markdown_cell("## 17. Error Analysis (Top Misclassifications)"))
    nb.cells.append(nbf.v4.new_code_cell(
'''misclassified.sort(key=lambda x: x['confidence'], reverse=True)
top_errors = misclassified[:20]

if len(top_errors) > 0:
    cols = min(5, len(top_errors))
    rows = (len(top_errors) + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    
    for i, err in enumerate(top_errors):
        ax = plt.subplot(rows, cols, i + 1)
        img = cv2.cvtColor(cv2.imread(err['path']), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"True: {err['true_label'].replace('coffee___', '')}\\nPred: {err['pred_label'].replace('coffee___', '')} ({err['confidence']:.2f})", fontsize=8, color='red')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(EXP_DIR / 'misclassified_images.png')
    plt.show()
'''))

    # 18 & 19. Model Complexity & Save Outputs
    nb.cells.append(nbf.v4.new_markdown_cell("## 18 & 19. Save Metrics & Outputs"))
    nb.cells.append(nbf.v4.new_code_cell(
'''total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

metrics = {
    "model_name": MODEL_NAME,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "training_time_sec": train_time,
    "inference_fps": fps,
    "best_val_loss": best_val_loss,
    "test_accuracy": report_dict['accuracy']
}

with open(EXP_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
    
with open(EXP_DIR / 'classification_report.json', 'w') as f:
    json.dump(report_dict, f, indent=4)

pd.DataFrame(history).to_csv(EXP_DIR / 'training_history.csv', index=False)
print("All outputs saved to experiment folder.")
'''))

    # 20 & 21. Sprint Report & Status
    nb.cells.append(nbf.v4.new_markdown_cell("## 20 & 21. Generate Report & Update Status"))
    nb.cells.append(nbf.v4.new_code_cell(
'''with open('../reports/Sprint_05_Report.md', 'w') as f:
    f.write(f"# Sprint 5: {MODEL_NAME} Baseline Report\\n\\n")
    f.write(f"## Configuration\\n- **Experiment Path**: `{EXP_DIR}`\\n- **Epochs**: {config['epochs']}\\n- **Optimizer**: {config['optimizer']}\\n")
    f.write(f"\\n## Model Complexity\\n- **Total Parameters**: {total_params:,}\\n- **Trainable Parameters**: {trainable_params:,}\\n- **Inference Speed**: {fps:.1f} FPS\\n")
    f.write(f"\\n## Results\\n- **Test Accuracy**: {report_dict['accuracy'] * 100:.2f}%\\n")
    f.write("\\nSee `results/mobilenetv3/experiment_XXX/` for confusion matrix and misclassifications.\\n")

status_file = Path('../data/reports/status.json')
with open(status_file, 'r') as f:
    status = json.load(f)

status['mobilenet'] = True
with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)
    
print("Sprint 5 Execution Completed.")
'''))

    Path('notebooks').mkdir(exist_ok=True)
    with open('notebooks/04_Baseline_Model_Framework.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
