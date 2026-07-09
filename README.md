# Coffee Leaf Disease Research

Automated machine learning pipeline for detecting diseases in coffee leaves using the RoCoLe dataset.

## Project Status

- [x] **Sprint 1:** Environment Setup & Pipeline Architecture
- [x] **Sprint 1.5:** Data Ingestion & Automated Verification (`data/reports/`)
- [x] **Sprint 2:** Exploratory Data Analysis (EDA) & Insights
- [x] **Sprint 3:** Image Preprocessing & Dataset Preparation
- [ ] **Sprint 4:** Data Augmentation
- [ ] **Sprint 4+:** Modeling & Evaluation

## Repository Structure

```text
Coffee-Leaf-Disease-Research/
│
├── datasets/                  # Source code for downloading and verifying data
│   ├── downloader.py          # Orchestrates download & analysis
│   └── utils.py               # Data verification & stats logic
│
├── scripts/                   # Auxiliary scripts
│   └── build_eda_notebook.py  # Autogenerates the EDA Jupyter notebook
│
├── data/                      
│   ├── raw/                   # Raw RoCoLe images (Ignored in Git)
│   ├── processed/             # Preprocessed images (Ignored in Git)
│   └── reports/               # Auto-generated dataset metrics, JSON stats, and EDA reports
│
├── plots/                     # Auto-generated plots from EDA (Histograms, Distributions, Outliers)
│
├── notebooks/                 # Jupyter Notebooks
│   └── 01_Dataset_Analysis.ipynb
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

2. **Download & Verify Data:**
   Make sure your `KAGGLE_API_TOKEN` is set, then run:
   ```bash
   python datasets/downloader.py
   ```

3. **Run Exploratory Data Analysis (EDA):**
   Build the notebook and run it:
   ```bash
   python scripts/build_eda_notebook.py
   jupyter nbconvert --to notebook --execute notebooks/01_Dataset_Analysis.ipynb
   ```

All plots will be output to `plots/` and statistical findings will be saved to `data/reports/`.