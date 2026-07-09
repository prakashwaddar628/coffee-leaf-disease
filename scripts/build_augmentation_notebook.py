import nbformat as nbf
from pathlib import Path

def create_notebook():
    nb = nbf.v4.new_notebook()

    # 1. Introduction
    nb.cells.append(nbf.v4.new_markdown_cell(
r"""# 1. Introduction
## Data Augmentation & PyTorch Data Pipeline

**Sprint Goal:** Build a production-quality PyTorch data pipeline that handles stratified splits, solves class imbalance, applies realistic augmentations, and yields a pipeline ready for model training.

**Research Questions:**
* RQ11: What is the best train/validation/test split? (Answer: 80/10/10)
* RQ12: Should augmentation be stored on disk? (Answer: No, on-the-fly via Albumentations to PyTorch Dataset)
* RQ13: Which augmentations preserve disease characteristics? (Horizontal Flip, Rotation, Random Brightness/Contrast, Gaussian Noise)
"""))

    # 2. Load Dataset
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Load Dataset"))
    nb.cells.append(nbf.v4.new_code_cell(
r"""import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

reports_dir = Path('../data/reports')
processed_dir = Path('../data/processed/final')

with open(reports_dir / 'status.json') as f:
    status = json.load(f)

with open('../datasets/config.yaml') as f:
    config = yaml.safe_load(f)['training']

print(f"Preprocessing Status: {'Complete' if status.get('preprocessing') else 'Incomplete'}")
print(f"Loaded Config: {config}")
"""))

    # 3. Train / Validation / Test Split
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Train / Validation / Test Split\nUsing stratified split to preserve class imbalance ratios."))
    nb.cells.append(nbf.v4.new_code_cell(
r"""from sklearn.model_selection import train_test_split
import shutil
import os

classes = [d.name for d in processed_dir.iterdir() if d.is_dir()]

# Gather all images and their labels
filepaths = []
labels = []
for cls in classes:
    for img_path in (processed_dir / cls).glob('*.jpg'):
        filepaths.append(str(img_path))
        labels.append(cls)

# Split 80 / 20
X_train, X_temp, y_train, y_temp = train_test_split(
    filepaths, labels, test_size=(1.0 - config['train_split']), stratify=labels, random_state=42
)

# Split remaining 20 into 10 / 10
val_ratio = config['validation_split'] / (config['validation_split'] + config['test_split'])
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(1.0 - val_ratio), stratify=y_temp, random_state=42
)

data_dir = Path('../data')
for split in ['train', 'validation', 'test']:
    for cls in classes:
        (data_dir / split / cls).mkdir(parents=True, exist_ok=True)

def copy_files(file_list, split_name):
    for f in file_list:
        src = Path(f)
        dst = data_dir / split_name / src.parent.name / src.name
        shutil.copy2(src, dst)

copy_files(X_train, 'train')
copy_files(X_val, 'validation')
copy_files(X_test, 'test')

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

split_stats = pd.DataFrame({
    'Split': ['Train', 'Validation', 'Test'],
    'Images': [len(X_train), len(X_val), len(X_test)]
})
split_stats.to_csv('../data/reports/split_statistics.csv', index=False)
"""))

    # 4. Class Imbalance
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Class Imbalance\nRecommendation: Keep original dataset, use WeightedRandomSampler."))
    nb.cells.append(nbf.v4.new_code_cell(
r"""from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler

class_counts = Counter(y_train)
total_samples = sum(class_counts.values())

class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

# Normalize weights for stability
weight_sum = sum(class_weights.values())
class_weights = {cls: w / weight_sum for cls, w in class_weights.items()}

print("Calculated Class Weights:")
for cls, w in class_weights.items():
    print(f"{cls}: {w:.4f}")

pd.DataFrame(list(class_weights.items()), columns=['Class', 'Weight']).to_csv('../data/reports/class_weights.csv', index=False)
"""))

    # 5. Albumentations Pipeline
    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Albumentations Pipeline\nBuilding realistic augmentations."))
    nb.cells.append(nbf.v4.new_code_cell(
r"""import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMG_SIZE = config['image_size']

# Note: Images are already CLAHE enhanced from Sprint 3. We focus on spatial and noise variations.
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# For visualization without normalization/tensors
vis_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=1.0),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=1.0, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
])
"""))

    # 6. Visualization
    nb.cells.append(nbf.v4.new_markdown_cell("## 6. Visualization\nVisualizing augmentations sequentially for thesis."))
    nb.cells.append(nbf.v4.new_code_cell(
r"""import matplotlib.pyplot as plt

plt.figure(figsize=(15, 3 * len(classes)))
headers = ["Original", "Horizontal Flip", "Rotate", "Brightness/Contrast", "Gauss Noise"]

for i, cls in enumerate(classes):
    img_path = next((data_dir / 'train' / cls).glob('*.jpg'))
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    
    # Generate sequential step transforms for visualization
    t1 = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])
    t2 = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0)])
    t3 = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.ShiftScaleRotate(rotate_limit=20, p=1.0)])
    t4 = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.ShiftScaleRotate(rotate_limit=20, p=1.0), A.RandomBrightnessContrast(p=1.0)])
    t5 = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.ShiftScaleRotate(rotate_limit=20, p=1.0), A.RandomBrightnessContrast(p=1.0), A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)])
    
    images = [
        t1(image=img)['image'], t2(image=img)['image'],
        t3(image=img)['image'], t4(image=img)['image'], t5(image=img)['image']
    ]
    
    for j, aug_img in enumerate(images):
        ax = plt.subplot(len(classes), 5, i * 5 + j + 1)
        ax.imshow(aug_img)
        if i == 0: ax.set_title(headers[j], fontweight='bold')
        if j == 0: ax.set_ylabel(cls.replace('coffee___', ''), rotation=90, size='large', fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
Path('../plots').mkdir(exist_ok=True)
plt.savefig('../plots/augmentation_comparison.png')
plt.show()
"""))

    # 7. PyTorch Dataset
    nb.cells.append(nbf.v4.new_markdown_cell("## 7. PyTorch Dataset\nImplementing `CoffeeLeafDataset`."))
    nb.cells.append(nbf.v4.new_code_cell(
r"""from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Callable, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoffeeLeafDataset(Dataset):
    '''
    PyTorch Dataset for Coffee Leaf Disease imagery.
    '''
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            for img_path in (self.root_dir / cls).glob('*.jpg'):
                self.samples.append((str(img_path), self.class_to_idx[cls]))
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                img = self.transform(image=img)['image']
                
            return img, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a zero tensor in case of corruption
            return torch.zeros((3, config['image_size'], config['image_size'])), label
"""))

    # 8. PyTorch DataLoader
    nb.cells.append(nbf.v4.new_markdown_cell("## 8. PyTorch DataLoader"))
    nb.cells.append(nbf.v4.new_code_cell(
r"""train_dataset = CoffeeLeafDataset(data_dir / 'train', transform=train_transform)
val_dataset = CoffeeLeafDataset(data_dir / 'validation', transform=val_transform)
test_dataset = CoffeeLeafDataset(data_dir / 'test', transform=val_transform)

# Configure WeightedRandomSampler for training
sample_weights = [class_weights[classes[label]] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(
    weights=sample_weights, 
    num_samples=len(sample_weights), 
    replacement=True
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    sampler=sampler, 
    num_workers=config['num_workers'],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=config['num_workers'],
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=config['num_workers'],
    pin_memory=True
)

print(f"DataLoaders initialized.\nTrain batches: {len(train_loader)}\nVal batches: {len(val_loader)}")
"""))

    # 9. Performance Benchmark
    nb.cells.append(nbf.v4.new_markdown_cell("## 9. Performance Benchmark"))
    nb.cells.append(nbf.v4.new_code_cell(
r"""import time

print("Benchmarking DataLoader throughput...")
start_time = time.time()

batches_to_test = min(5, len(train_loader))
count = 0

for i, (images, labels) in enumerate(train_loader):
    count += 1
    if count >= batches_to_test:
        break

end_time = time.time()
total_time = end_time - start_time
throughput = (batches_to_test * config['batch_size']) / total_time

print(f"Processed {batches_to_test} batches ({batches_to_test * config['batch_size']} images)")
print(f"Time taken: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} images / second")
"""))

    # 10. Visual Batch Inspection
    nb.cells.append(nbf.v4.new_markdown_cell("## 10. Visual Batch Inspection"))
    nb.cells.append(nbf.v4.new_code_cell(
r"""import torchvision

images, labels = next(iter(train_loader))

# Un-normalize for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
vis_images = images * std + mean
vis_images = torch.clamp(vis_images, 0, 1)

grid = torchvision.utils.make_grid(vis_images[:16], nrow=4)

plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0))
plt.title("Visual Batch Inspection (16 images)")
plt.axis('off')
plt.show()
"""))

    # 11. Generate Reports
    nb.cells.append(nbf.v4.new_markdown_cell("## 11. Generate Reports"))
    nb.cells.append(nbf.v4.new_code_cell(
r"""with open('../data/reports/augmentation_report.json', 'w') as f:
    json.dump({
        "throughput_img_per_sec": throughput,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset)
    }, f, indent=4)

with open('../data/reports/dataloader_configuration.json', 'w') as f:
    json.dump(config, f, indent=4)

with open('../data/reports/augmentation_summary.md', 'w') as f:
    f.write("# Data Augmentation Summary\n")
    f.write("- **Split Strategy**: 80/10/10 Stratified Split\n")
    f.write("- **Imbalance Resolution**: `WeightedRandomSampler`\n")
    f.write("- **Augmentations Used**: HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, GaussNoise\n")
    f.write("- **Rejected Augmentations**: RandomCrop, Elastic Transform (biologically destructive)\n")
"""))

    # 12. Update Status
    nb.cells.append(nbf.v4.new_markdown_cell("## 12. Update Status"))
    nb.cells.append(nbf.v4.new_code_cell(
r"""status['augmentation'] = True
with open('../data/reports/status.json', 'w') as f:
    json.dump(status, f, indent=4)

print("Sprint 4 Pipeline Complete. Ready for Model Training.")
"""))

    Path('notebooks').mkdir(exist_ok=True)
    with open('notebooks/03_Data_Augmentation.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
