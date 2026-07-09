# Coffee Leaf Disease Research

Automated machine learning pipeline for detecting diseases in coffee leaves using the RoCoLe dataset.

## Project Status

- [x] **Sprint 1:** Environment Setup & Pipeline Architecture
- [x] **Sprint 1.5:** Data Ingestion & Automated Verification (`data/reports/`)
- [x] **Sprint 2:** Exploratory Data Analysis (EDA) & Insights
- [x] **Sprint 3:** Image Preprocessing & Dataset Preparation
- [x] **Sprint 4:** Data Augmentation & PyTorch Data Pipeline
- [x] **Sprint 5:** Baseline Modeling & Evaluation

## Repository Structure

```text
Coffee-Leaf-Disease-Research/
│
├── datasets/                  # Configuration & Data Utils
│   ├── config.yaml            # Universal hyperparameters for all ML pipelines
│   ├── downloader.py          # Orchestrates Kaggle download & analysis
│   └── utils.py               # Data verification & stats logic
│
├── scripts/                   # Auxiliary notebook generators
│   ├── build_eda_notebook.py  
│   ├── build_preprocessing_notebook.py
│   ├── build_augmentation_notebook.py
│   └── build_training_notebook.py
│
├── data/                      
│   ├── raw/                   # Raw RoCoLe images (Ignored in Git)
│   ├── processed/             # Resized and CLAHE-enhanced images (Ignored in Git)
│   ├── train/                 # 80% Stratified Split (Ignored in Git)
│   ├── validation/            # 10% Stratified Split (Ignored in Git)
│   ├── test/                  # 10% Stratified Split (Ignored in Git)
│   └── reports/               # Auto-generated metrics, stats, & logic reports
│
├── plots/                     # Auto-generated plots (EDA, Preprocessing, Augmentation)
│
├── notebooks/                 # Sequential ML Pipeline Notebooks
│   ├── 01_Dataset_Analysis.ipynb
│   ├── 02_Image_Preprocessing.ipynb
│   ├── 03_Data_Augmentation.ipynb
│   └── 04_Baseline_Model_Framework.ipynb
│
├── results/                   # Experiment Tracking (Immutable runs)
│   └── mobilenetv3/           # Model-specific folder
│       └── experiment_001/    # Checkpoints, JSON metrics, and performance plots
│
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore
```

## Getting Started

1. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Execute Full Pipeline (Sprints 1-4):**
   Make sure your `KAGGLE_API_TOKEN` is set, then sequentially execute:
   ```bash
   # Sprint 1.5: Ingestion
   python datasets/downloader.py
   
   # Sprint 2: Exploratory Data Analysis
   python scripts/build_eda_notebook.py
   jupyter nbconvert --to notebook --execute notebooks/01_Dataset_Analysis.ipynb

   # Sprint 3: Image Preprocessing (Resize + CLAHE)
   python scripts/build_preprocessing_notebook.py
   jupyter nbconvert --to notebook --execute notebooks/02_Image_Preprocessing.ipynb

   # Sprint 4: Data Augmentation & Train/Val Splits
   python scripts/build_augmentation_notebook.py
   jupyter nbconvert --to notebook --execute notebooks/03_Data_Augmentation.ipynb

   # Sprint 5: Universal Training Loop & MobileNetV3 Baseline
   python scripts/build_training_notebook.py
   jupyter nbconvert --to notebook --execute notebooks/04_Baseline_Model_Framework.ipynb
   ```

All visual output will be saved to `plots/` and all empirical stats will be written directly into `data/reports/`.