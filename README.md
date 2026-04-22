# Exoplanet Candidate Vetting Using a Simplified Genesis CNN

**MSc Data Science — University of Hertfordshire**
**Module: 7PAM2002 | Student: Mohammad Aakash | Student ID: 24079227**
**Supervisor: Aoife Simpson**

---

## Project Overview

This project applies machine learning to automatically classify Kepler Objects of Interest (KOIs) from NASA's Kepler Space Telescope into three categories: **CONFIRMED**, **FALSE POSITIVE**, and **CANDIDATE**. Rather than processing raw photometric light curves, the project works directly with 37 pre-extracted astrophysical features from the NASA Exoplanet Archive (Q1–Q17 DR25).

Five models were trained and evaluated on the same dataset and held-out test set:

| Model | Accuracy | F1 Macro | ROC-AUC |
|---|---|---|---|
| Random Forest | 89.4% | 0.778 | 0.955 |
| Genesis CNN | 85.7% | 0.749 | 0.879 |
| Logistic Regression | 81.9% | 0.763 | 0.924 |
| SVM (RBF) | 81.9% | 0.739 | 0.897 |
| Baseline CNN | 80.7% | 0.738 | 0.908 |

The **Genesis CNN** is the primary contribution — a dual-branch Conv1D architecture where the local branch (kernel=3) captures adjacent feature interactions and the global branch (kernel=7, 5) captures wider cross-group patterns across the 37-feature sequence.

---

## Dataset

- **Source:** NASA Exoplanet Archive — Kepler KOI Cumulative Catalogue (Q1–Q17 DR25)
- **Download:** https://exoplanetarchive.ipac.caltech.edu/
- **File:** `koi_data.csv`
- **Size:** 3,901 rows × 141 columns (raw) → 3,900 rows × 37 features (after preprocessing)
- **Target:** `koi_disposition` — CONFIRMED / FALSE POSITIVE / CANDIDATE


---

## Project Structure

```
exoplanet-vetting/
│
├── data/
│   └── raw/
│       └── koi_data.csv          
│
├── src/
│   ├── utils/
│   │   └── config.py             ← feature lists, model parameters, file paths
│   │
│   ├── data/
│   │   ├── preprocess.py         ← 6-stage preprocessing pipeline
│   │   └── run_eda.py            ← EDA figure generation
│   │
│   ├── models/
│   │   ├── classical_ml.py       ← Random Forest, SVM, Logistic Regression
│   │   ├── baseline_cnn.py       ← single-branch Baseline CNN
│   │   └── genesis_cnn.py        ← dual-branch Genesis CNN (primary model)
│   │
│   └── evaluation/
│       ├── metrics.py            ← shared evaluation functions
│       └── compare_models.py     ← cross-model comparison and visualisation
│
├── results/
│   ├── figures/                  ← all generated plots and figures
│   └── metrics/                  ← CSV files with model performance results
│
├── requirements.txt
├── run_pipeline.py               ← master script to run everything end to end
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.9 or higher

**1. Clone the repository**
```bash
git clone https://github.com/[your-username]/exoplanet-vetting.git
cd exoplanet-vetting
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```


## Dependencies

```
numpy
pandas
scikit-learn
tensorflow>=2.10
keras
matplotlib
seaborn
scipy
```

All dependencies are listed in `requirements.txt`.

---

## Running the Project

**Run the full pipeline end to end:**
```bash
python run_pipeline.py
```

This will run all six stages in order:
1. Preprocessing
2. EDA and figure generation
3. Classical ML training and evaluation
4. Baseline CNN training and evaluation
5. Genesis CNN training and evaluation
6. Cross-model comparison

**Run individual stages:**
```bash
# Preprocessing only
python -m src.data.preprocess

# EDA only
python -m src.data.run_eda

# Classical models only
python -m src.models.classical_ml

# Baseline CNN only
python -m src.models.baseline_cnn

# Genesis CNN only
python -m src.models.genesis_cnn

# Compare all models
python -m src.evaluation.compare_models
```

---

## Preprocessing Pipeline

The preprocessing follows a strict no-leakage protocol — all transformers are fitted on training data only and applied to validation and test sets.

| Stage | Description |
|---|---|
| 1. Load | CSV loaded with `comment='#'` to skip NASA header lines |
| 2. Drop columns | 104 columns removed — 20 empty, 17 identifiers, 2 leaky |
| 3. Encode labels | CONFIRMED→0, FALSE POSITIVE→1, CANDIDATE→2 |
| 4. Stratified split | 70/15/15 train/val/test, `random_state=42` |
| 5. Median imputation | Fitted on train only, max 5.4% missing in any feature |
| 6. StandardScaler | Fitted on train only, zero mean and unit variance |

> **Note:** `koi_score` and `koi_pdisposition` are explicitly excluded as they are direct outputs of the NASA Robovetter and constitute data leakage.

---

## Model Architectures

### Genesis CNN (Primary Model)
```
Input (37 features, 1 channel)
         │
    ┌────┴────┐
    │         │
LOCAL         GLOBAL
kernel=3,3    kernel=7,5
filters=32,64 filters=32,64
    │         │
   GAP        GAP
  (64,)      (64,)
    │         │
    └────┬────┘
    Concatenate (128,)
         │
   Dense(128, ReLU) + Dropout(0.4)
   Dense(64, ReLU)  + Dropout(0.4)
   Dense(3, Softmax)

Total parameters: 42,435
```

### Baseline CNN (Reference Model)
```
Input → Conv1D(32,k=3) → Conv1D(64,k=3) → Conv1D(128,k=3)
     → GlobalAveragePooling → Dense(64) → Dropout(0.3) → Dense(3, Softmax)

Total parameters: 40,163
```

---

## Results

All results are saved to `results/metrics/` as CSV files:

- `classical_ml_results.csv` — Random Forest, SVM, Logistic Regression
- `baseline_cnn_results.csv` — Baseline CNN per-epoch and final metrics
- `genesis_cnn_results.csv` — Genesis CNN per-epoch and final metrics
- `all_models_metrics.csv` — Combined comparison across all five models

All figures are saved to `results/figures/`.

---

## Key Findings

- **Random Forest** outperforms all models (89.4% accuracy) — tree-based ensembles are better suited to small structured tabular datasets
- **Genesis CNN** validates the dual-branch design — outperforms Baseline CNN on all primary metrics and improves CONFIRMED F1 from 0.842 to 0.890
- **CANDIDATE class** is universally the hardest to classify (F1 range 0.379–0.461) — reflecting the inherent ambiguity of this label in the astronomical literature
- **Data size** is the primary limitation — augmentation experiments show Genesis CNN accuracy reaches ~87–88% when training data is doubled, closing the gap with Random Forest

---

## Future Work

- Apply Genesis CNN to raw Kepler flux time series (binary classification on 3,197 flux measurements per star)
- Apply SMOTE oversampling to improve CANDIDATE class F1
- Replace GlobalAveragePooling with a self-attention mechanism
- Ensemble Genesis CNN probability outputs with Random Forest predictions

---

## References

- Visser, K., Bosma, B., & Postma, E. (2022). Exoplanet detection with Genesis. *Journal of Astronomical Instrumentation, 11*(3). https://doi.org/10.1142/S2251171722500118
- Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning. *The Astronomical Journal, 155*(2), 94. https://doi.org/10.3847/1538-3881/aa9e09
- Thompson, S. E., et al. (2018). Planetary Candidates Observed by Kepler. VIII. *ApJS, 235*(2), 38. https://doi.org/10.3847/1538-4365/aab4f9
- Akeson, R. L., et al. (2013). The NASA Exoplanet Archive. *PASP, 125*(930), 989. https://doi.org/10.1086/672273

---

## License

This project was developed for academic purposes as part of an MSc Data Science dissertation at the University of Hertfordshire. The dataset is sourced from NASA's publicly available Exoplanet Archive and is subject to NASA's open data policy.