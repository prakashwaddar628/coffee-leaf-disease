"""
Standalone ResNet50 training script.
Equivalent to notebooks/05_Model_ResNet50.ipynb but runs without nbconvert timeout.
"""
import yaml
import json
import os
import time
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ── Config ────────────────────────────────────────────────────────────────────
with open('datasets/config.yaml') as f:
    config = yaml.safe_load(f)['training']

MODEL_NAME = "resnet50"
RESULTS_DIR = Path(f"results/{MODEL_NAME}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

existing_exp = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("experiment_")]
exp_num = len(existing_exp) + 1
EXP_DIR = RESULTS_DIR / f"experiment_{exp_num:03d}"
EXP_DIR.mkdir(parents=True, exist_ok=True)

with open(EXP_DIR / "config_snapshot.yaml", 'w') as f:
    yaml.dump(config, f)

print(f"Experiment: {EXP_DIR}")
print(f"Config: {config}")

# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config['random_seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Dataset ────────────────────────────────────────────────────────────────────
data_dir = Path('data')
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
    def __init__(self, root_dir, transform=None):
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
val_dataset   = CoffeeLeafDataset(data_dir / 'validation', transform=val_transform)
test_dataset  = CoffeeLeafDataset(data_dir / 'test', transform=val_transform)
classes = train_dataset.classes

class_weights_df = pd.read_csv('data/reports/class_weights.csv')
class_weights_dict = dict(zip(class_weights_df['Class'], class_weights_df['Weight']))
sample_weights = [class_weights_dict[classes[label]] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=config['num_workers'])
val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False,  num_workers=config['num_workers'])
test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'], shuffle=False,  num_workers=config['num_workers'])

print(f"Classes: {classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ── Model ──────────────────────────────────────────────────────────────────────
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)
print("ResNet50 loaded. Backbone frozen.")

# ── Training Components ────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
patience  = config['early_stopping']

# ── Training Loop ──────────────────────────────────────────────────────────────
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

best_val_loss = float('inf')
epochs_no_improve = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

start_time = time.time()
print(f"\nStarting training for {config['epochs']} epochs...")
for epoch in range(config['epochs']):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc     = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), EXP_DIR / "best_model.pth")
    else:
        epochs_no_improve += 1

    print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if epochs_no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}!")
        break

torch.save(model.state_dict(), EXP_DIR / "last_model.pth")
train_time = time.time() - start_time
print(f"\nTraining completed in {train_time:.2f}s")

# ── Evaluation ─────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(EXP_DIR / "best_model.pth"))
model.eval()

all_preds, all_labels, all_probs, misclassified = [], [], [], []

infer_start = time.time()
with torch.no_grad():
    for images, labels, paths in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified.append({
                    'path': paths[i],
                    'true_label': classes[labels[i]],
                    'pred_label': classes[preds[i]],
                    'confidence': probs[i][preds[i]].item()
                })

fps = len(test_dataset) / (time.time() - infer_start)
report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
print(classification_report(all_labels, all_preds, target_names=classes))

# ── Visualizations ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_title('Loss Curve'); ax1.legend()
ax2.plot(history['train_acc'], label='Train Acc')
ax2.plot(history['val_acc'], label='Val Acc')
ax2.set_title('Accuracy Curve'); ax2.legend()
plt.savefig(EXP_DIR / 'training_curves.png')
plt.close()

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(EXP_DIR / 'confusion_matrix.png')
plt.close()

# ── Save Metrics ───────────────────────────────────────────────────────────────
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_mb    = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

metrics = {
    "model_name": MODEL_NAME,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "model_size_mb": model_size_mb,
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

print(f"\nTest Accuracy: {report_dict['accuracy']*100:.2f}%")
print(f"Inference FPS: {fps:.1f}")
print(f"Model Size: {model_size_mb:.2f} MB")
print(f"All outputs saved to {EXP_DIR}")

# ── Sprint Report ──────────────────────────────────────────────────────────────
with open('reports/Sprint_06_Report.md', 'w') as f:
    f.write(f"# Sprint 6: {MODEL_NAME} Baseline Report (3k Dataset)\n\n")
    f.write(f"## Objective\nEvaluate ResNet50 on the 3,000-image harmonized dataset (RoCoLe + JMuBEN).\n\n")
    f.write(f"## Configuration\n- **Experiment**: `{EXP_DIR}`\n- **Epochs**: {config['epochs']}\n- **Optimizer**: {config['optimizer']}\n\n")
    f.write(f"## Model Complexity\n- **Total Parameters**: {total_params:,}\n- **Trainable Parameters**: {trainable_params:,}\n- **Model Size**: {model_size_mb:.2f} MB\n- **Inference Speed**: {fps:.1f} FPS\n\n")
    f.write(f"## Results\n- **Test Accuracy**: {report_dict['accuracy']*100:.2f}%\n\n")
    f.write(f"## Per-Class Performance\n")
    for cls in classes:
        r = report_dict.get(cls, {})
        f.write(f"- **{cls}**: P={r.get('precision',0):.3f}, R={r.get('recall',0):.3f}, F1={r.get('f1-score',0):.3f}\n")

status_file = Path('data/reports/status.json')
with open(status_file, 'r') as f:
    status = json.load(f)
status['resnet'] = True
with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)

print("\nSprint 6 complete!")
