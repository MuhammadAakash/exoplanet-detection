# Stage 3 — Classical ML: Summary & Quick Reference
> Exoplanet Candidate Vetting · MSc Data Science Dissertation  
> A compact revision card covering all three models and key decisions

---

## The Three Models at a Glance

| | Random Forest | SVM | Logistic Regression |
|--|:---:|:---:|:---:|
| **Type** | Ensemble (300 trees) | Kernel method | Linear model |
| **Boundary** | Non-linear (step functions) | Non-linear (RBF kernel) | Linear |
| **Needs StandardScaler?** | ✗ No | ✓ Yes | ✓ Yes |
| **Interpretable?** | ✗ Black box | ✗ Black box | ✓ Coefficients readable |
| **Probability calibration** | Good (vote fractions) | Poor (Platt scaling) | Best (log-loss direct) |
| **Handles imbalance** | `class_weight='balanced'` | `class_weight='balanced'` | `class_weight='balanced'` |

---

## Results in One Table

| Model | Accuracy | F1 Macro | ROC-AUC | κ | F1 CONF | F1 FP | F1 CAND |
|-------|:--------:|:--------:|:-------:|:-:|:-------:|:-----:|:-------:|
| **RF** | **89.4%** | **0.778** | **0.955** | **0.796** | **0.92** | 0.98 | 0.43 |
| LR | 81.9% | 0.763 | 0.924 | 0.692 | 0.84 | 0.99 | **0.46** |
| SVM | 81.9% | 0.739 | 0.897 | 0.681 | 0.85 | **0.99** | 0.38 |

**Bold = best for that metric.** RF wins everything except CANDIDATE F1 (LR wins) and FALSE POSITIVE F1 (LR and SVM tie).

---

## Key Parameters (all models)

### Most important parameter across all three models
```python
class_weight = 'balanced'
```
- Dataset: 2,341 CONFIRMED vs 477 CANDIDATE (4.9× imbalance)
- Without balancing: CANDIDATE F1 drops to 0.08–0.15 for all models
- With balancing: CANDIDATE samples weighted ~4.9× more during training

### Random Forest
```python
RandomForestClassifier(
    n_estimators=300,      # 300 trees — diminishing returns past ~200
    max_depth=None,        # Full depth — ensemble corrects overfitting
    max_features='sqrt',   # √37 ≈ 6 features per split — forces tree diversity
    class_weight='balanced',
    random_state=42,
    n_jobs=-1              # Parallel training — no effect on predictions
)
```

### SVM
```python
SVC(
    kernel='rbf',          # Non-linear. K(x,x') = exp(−γ||x−x'||²)
    C=10,                  # Moderate-strict. Low C=underfit, High C=overfit
    gamma='scale',         # Auto-calibrated: 1/(n_features × variance)
    probability=True,      # Platt scaling — needed for ROC-AUC
    class_weight='balanced',
    random_state=42
)
```

### Logistic Regression
```python
LogisticRegression(
    C=1.0,                 # Standard L2 regularisation
    max_iter=1000,         # Default 100 is too few for this problem
    class_weight='balanced',
    solver='lbfgs',        # Multinomial softmax formulation
    penalty='l2',          # Ridge — all features kept
    random_state=42
)
```

---

## How Each Model Makes a Prediction

### Random Forest
```
Input: 37 features → 300 decision trees vote → majority class wins
       → class with most votes = prediction
       → vote fraction = probability
```

### SVM
```
Input: 37 features (StandardScaled) → 3 binary SVMs (OvR)
       → each SVM scores the sample → class with highest score wins
       → Platt scaling converts scores to probabilities
```

### Logistic Regression
```
Input: 37 features (StandardScaled) → multiply by weight matrix (3×37)
       → 3 raw scores → softmax → 3 probabilities summing to 1
       → argmax = prediction
```

---

## Why RF Wins

1. **Non-linear boundaries** → learns `koi_prad > 15 → FP` as a single split
2. **Variance reduction** → 300 trees' errors cancel in the vote
3. **Feature diversity** → random subsets force trees to learn different patterns
4. **No assumptions** → trees make no distributional assumptions about features

## Why LR Beats SVM on CANDIDATE F1

LR's soft linear boundary with balanced weights produces:
- **Higher recall** (0.56 vs SVM 0.34) — finds more actual candidates
- **Lower precision** (0.39 vs SVM 0.43) — more false alarms
- **Higher F1** (0.46 vs SVM 0.38) — harmonic mean balances both

SVM's max-margin objective prioritises separating the two most distinct classes (CONFIRMED and FALSE POSITIVE), leaving CANDIDATE with a weaker boundary. LR cannot be aggressive in the same way — it accidentally helps CANDIDATE recall.

## Why CANDIDATE is Hard for All Models

CANDIDATE is not a physical class — it is an administrative label meaning "unresolved". Some are real planets, some are false positives. Their features are identical to both other classes. No model can separate signals when the ground truth is genuinely unknown.

**This is a data limitation, not a modelling failure.**

---

## Metric Quick Reference

| Metric | Formula | Range | Good value |
|--------|---------|-------|------------|
| Accuracy | correct / total | 0–1 | Misleading alone for imbalanced data |
| Precision | TP / (TP+FP) | 0–1 | High = few false alarms |
| Recall | TP / (TP+FN) | 0–1 | High = catches most real members |
| F1 | 2×P×R/(P+R) | 0–1 | >0.8 = strong |
| F1 Macro | avg F1 per class | 0–1 | Equal weight to all classes |
| ROC-AUC | area under ROC | 0.5–1 | >0.9 = strong |
| Cohen's κ | above-chance agree | 0–1 | 0.6–0.8 = substantial |

---

## Files Generated in Stage 3

```
results/
├── figures/
│   ├── ml_01_confusion_matrices.png    ← 3×3 normalised CMs for all models
│   ├── ml_02_per_class_f1.png          ← F1 heatmap across models & classes
│   ├── ml_03_roc_curves.png            ← OvR ROC curves for all models
│   ├── ml_04_rf_feature_importance.png ← Top 20 RF Gini importances
│   └── ml_05_model_comparison.png      ← Grouped bar chart — 4 metrics × 3 models
├── metrics/
│   ├── classical_ml_results.csv        ← Summary row per model
│   ├── random_forest_metrics.json      ← Full metrics dict for RF
│   ├── svm_metrics.json
│   ├── logistic_regression_metrics.json
│   └── all_models_metrics.csv          ← Master CSV (updated by each model)
└── models/
    ├── random_forest.pkl
    ├── svm.pkl
    └── logistic_regression.pkl
```

---

## What's Next: Stage 4 — Baseline CNN

Stage 3 established:
- **RF** = best classical ML baseline (accuracy 89.4%, F1 0.778, AUC 0.955)
- **LR** = best CANDIDATE F1 (0.46) — important edge case result
- **SVM** = best FALSE POSITIVE F1 (0.99) — theoretically clean boundaries

Stage 4 builds a **single-branch Conv1D CNN** on the same 37 tabular features. It must be evaluated against the RF baseline.

Stage 5 builds the **dual-branch Genesis CNN** (local kernel + global kernel branches). The core dissertation contribution.

Stage 6 compares all 5 models (RF, SVM, LR, Baseline CNN, Genesis CNN) on the same test set.

---

*Quick reference notes for Stage 3 — Classical ML*  
*Full detail: `ML_notes_random_forest.md`, `ML_notes_svm.md`, `ML_notes_logistic_regression.md`, `ML_notes_metrics_explained.md`*
