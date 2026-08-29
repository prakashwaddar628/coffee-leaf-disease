# Coffee Leaf Disease Research

Automated machine learning pipeline for detecting and classifying diseases in coffee leaves using a harmonized multi-source dataset (RoCoLe + JMuBEN).

## Project Status

- [x] **Sprint 1:** Environment Setup & Pipeline Architecture
- [x] **Sprint 1.5:** Data Ingestion & Automated Verification (`data/reports/`)
- [x] **Sprint 2:** Exploratory Data Analysis (EDA) & Insights
- [x] **Sprint 3:** Image Preprocessing & Dataset Preparation
- [x] **Sprint 4:** Data Augmentation & PyTorch Data Pipeline
- [x] **Sprint 5:** Baseline Modeling & Evaluation (MobileNetV3)
- [x] **Sprint 6:** ResNet50 Baseline Evaluation
- [x] **Sprint 6.1:** Dataset Expansion & Harmonization (RoCoLe + JMuBEN → 3,000 images, 6 classes)
- [ ] **Dataset rebuild:** required before any reported model metric is treated as valid. Earlier split directories retained stale files and have been superseded by the verified rebuild scripts.

## Dataset

| Source | Images | Classes |
|:---|---:|:---|
| **RoCoLe** (Robusta Coffee Leaf) | 1,560 | `healthy`, `rust`, `red_spider_mite` |
| **JMuBEN** (Arabica Coffee Leaf) | 58,549 | `healthy`, `rust`, `cercospora`, `miner`, `phoma` |
| **Harmonized Research Subset** | **3,000** | All 6 classes combined |

The harmonized dataset is built by the `scripts/build_harmonized_dataset.py` script which merges both sources into a unified 6-class taxonomy, sampling from JMuBEN to reach the 3,000-image target.

## Model Comparison (3,000-image Harmonized Dataset)

| Metric | MobileNetV3 (Frozen) | MobileNetV3 (Fine-Tuned) | ResNet50 (Frozen) |
|:---|:---|:---|:---|
| **Accuracy** | — | **Historical: 84.1% (requires verified rebuild)** | **Historical 3-class run (not comparable)** |
| **Inference FPS** | — | 48.5 | ⏳ Training |
| **Total Parameters** | — | 1,524,006 | 23,514,179 |
| **Training Time** | — | ~953s | ⏳ Training |

> Previous results on the 1,560-image RoCoLe-only dataset are archived in `reports/Model_Comparison.md`.

## Repository Structure

```text
Coffee-Leaf-Disease-Research/
│
├── datasets/                        # Configuration & Data Utils
│   ├── config.yaml                  # Universal hyperparameters for all ML pipelines
│   ├── dataset.yaml                 # Dataset registry (RoCoLe, JMuBEN, BRACOL)
│   ├── downloader.py                # Orchestrates Kaggle download & analysis
│   └── utils.py                     # Data verification & stats logic
│
├── scripts/                         # Notebook generators & standalone training scripts
│   ├── build_eda_notebook.py
│   ├── build_preprocessing_notebook.py
│   ├── build_augmentation_notebook.py
│   ├── build_training_notebook.py
│   ├── build_resnet_notebook.py
│   ├── build_harmonized_dataset.py  # Merges RoCoLe + JMuBEN → 3,000-image corpus
│   └── train_resnet50.py            # Standalone ResNet50 trainer (no timeout)
│
├── data/
│   ├── raw/
│   │   ├── rocole/                  # Raw RoCoLe images (Ignored in Git)
│   │   ├── jmuben/                  # Raw JMuBEN images (Ignored in Git)
│   │   └── harmonized/              # Unified 3,000-image, 6-class dataset (Ignored in Git)
│   ├── processed/                   # Resized and CLAHE-enhanced images (Ignored in Git)
│   ├── train/                       # 80% Stratified Split (Ignored in Git)
│   ├── validation/                  # 10% Stratified Split (Ignored in Git)
│   ├── test/                        # 10% Stratified Split (Ignored in Git)
│   └── reports/                     # Auto-generated metrics, stats, & logic reports
│
├── plots/                           # Auto-generated plots (EDA, Preprocessing, Augmentation)
│
├── notebooks/                       # Sequential ML Pipeline Notebooks
│   ├── 01_Dataset_Analysis.ipynb
│   ├── 02_Image_Preprocessing.ipynb
│   ├── 03_Data_Augmentation.ipynb
│   ├── 04_Baseline_Model_Framework.ipynb
│   └── 05_Model_ResNet50.ipynb
│
├── results/                         # Experiment Tracking (Immutable runs)
│   ├── mobilenetv3/
│   │   └── experiment_XXX/          # Checkpoints, JSON metrics, plots
│   └── resnet50/
│       └── experiment_XXX/
│
├── reports/                         # Research reports
│   ├── Model_Comparison.md          # Architecture benchmark table
│   └── Sprint_06_Report.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started

1. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download Datasets (requires Kaggle API token):**
   ```bash
   python datasets/downloader.py --download rocole
   python datasets/downloader.py --download jmuben
   ```

3. **Build Harmonized 3,000-image Dataset:**
   ```bash
   python scripts/build_harmonized_dataset.py --clean
   ```

4. **Execute Full Pipeline:**
   ```bash
   # Sprint 2: EDA
   python scripts/build_eda_notebook.py
   jupyter nbconvert --to notebook --execute notebooks/01_Dataset_Analysis.ipynb --inplace

   # Sprint 3: Preprocessing (Resize + CLAHE)
   python scripts/build_preprocessing_notebook.py
   jupyter nbconvert --to notebook --execute notebooks/02_Image_Preprocessing.ipynb --inplace

   # Sprint 4: Augmentation & Train/Val/Test Split
   python scripts/build_augmentation_notebook.py
   jupyter nbconvert --to notebook --execute notebooks/03_Data_Augmentation.ipynb --inplace

   # Preferred: create a clean, deterministic split manifest and verify there
   # are no duplicate images across train, validation, and test.
   python scripts/build_dataset_splits.py --clean

   # Sprint 5: MobileNetV3 Baseline
   python scripts/build_training_notebook.py
   jupyter nbconvert --to notebook --execute notebooks/04_Baseline_Model_Framework.ipynb --inplace

   # Sprint 6: ResNet50 (use standalone script to avoid timeout)
   python scripts/train_resnet50.py
   ```

All visual output is saved to `plots/` and all empirical stats are written to `data/reports/`. Experiment metrics are tracked under `results/<model_name>/experiment_XXX/`.

> Use Python 3.11+ with a newly created virtual environment. The committed `venv/` directory is machine-specific and may not work on another machine. Recreate it with `py -m venv venv` (or your installed Python command), then install `requirements.txt`.
