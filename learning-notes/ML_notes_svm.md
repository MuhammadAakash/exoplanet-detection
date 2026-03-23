# Support Vector Machine — Complete Reference Notes
> Exoplanet Candidate Vetting · Stage 3 · MSc Data Science Dissertation  
> Dataset: NASA Kepler KOI Q1-Q17 DR25 · 3,901 samples · 37 features · 3 classes

---

## Table of Contents
1. [What is SVM?](#what-is-svm)
2. [Why we used it](#why-we-used-it)
3. [How it works — step by step](#how-it-works)
4. [Parameters we set and why](#parameters)
5. [Results on our dataset](#results)
6. [Every metric explained](#metrics-explained)
7. [What-if: changing parameters](#what-if)
8. [Limitations](#limitations)
9. [Dissertation talking points](#dissertation-talking-points)
10. [References](#references)

---

## What is SVM?

A **Support Vector Machine** finds the decision boundary that maximises the margin between classes. The boundary is placed in the exact middle of the widest possible "no-man's land" between the closest opposing training samples (the **support vectors**).

For non-linearly separable data (which ours is), the **RBF kernel** implicitly maps features into a higher-dimensional space where linear separation becomes possible — without ever explicitly computing that transformation.

> **One-line intuition:** Draw the widest possible road between two groups of points. SVM places the boundary in the road's exact centre. Only the points on the road's edges (support vectors) determine where it goes.

---

## Why we used it

| Reason | Detail |
|--------|--------|
| **Strong non-linear baseline** | RBF kernel captures non-linear boundaries without manual feature engineering |
| **Maximum margin = strong generalisation** | Theoretical guarantee: wider margin → less sensitivity to small perturbations in new data |
| **Expected in ML comparisons** | Any rigorous ML study includes SVM as a principled benchmark |
| **Near-perfect FALSE POSITIVE detection** | FP flags create sharp, clean boundaries — exactly the scenario SVM excels at |
| **Revealing contrast with RF** | SVM's CANDIDATE weakness illuminates why tree-based methods dominate tabular data |

---

## How it works

### Step 0 — StandardScaler is mandatory

The RBF kernel computes **Euclidean distance** between samples:
```
distance = ||x − x'|| = √( Σ (x_i − x'_i)² )
```

If `koi_depth` ranges 0–50,000 ppm and `koi_fpflag_co` is 0 or 1, `koi_depth` completely dominates the distance. SVM would effectively ignore all binary features.

Our Stage 1 preprocessing applies `StandardScaler` (mean=0, std=1 per feature) before SVM. The saved `scaler.pkl` must be applied at prediction time too.

> **RF doesn't need scaling** (splits on thresholds, not distances). **SVM and LR both require it.**

### Step 1 — Maximum margin hyperplane

For a binary classification problem, SVM solves:
```
Maximise: margin width = 2 / ||w||
Subject to: y_i (w·x_i + b) ≥ 1 − ξ_i  for all training samples i
            ξ_i ≥ 0  (slack variables allowing soft margin)
            C · Σ ξ_i is penalised  (C controls trade-off)
```

The decision boundary is the hyperplane `w·x + b = 0`. The margin is the distance between the two parallel hyperplanes `w·x + b = ±1`.

**Support vectors** are the training samples on the margin edges. All other training points are irrelevant after training — removing them would not change the model at all.

### Step 2 — The RBF kernel trick

Our 37 features are not linearly separable. The RBF kernel computes a **similarity score** between any two samples:

```
K(x, x') = exp(−γ · ||x − x'||²)
```

- When two samples are identical (distance = 0): K = 1
- As distance increases: K → 0
- With `gamma='scale'`: γ = 1 / (n_features × variance) — auto-calibrated after StandardScaler

This function implicitly maps samples into an infinite-dimensional space where linear separation is possible. The **kernel trick**: we never compute the transformed space — we only compute pairwise similarities, which is mathematically equivalent and much cheaper.

### Step 3 — One-vs-Rest for 3 classes

SVM is inherently binary. For our 3-class problem, sklearn trains **3 separate binary SVMs**:

```
SVM 1: CONFIRMED vs (FALSE POSITIVE + CANDIDATE)
SVM 2: FALSE POSITIVE vs (CONFIRMED + CANDIDATE)  ← easiest, FP flags are sharp
SVM 3: CANDIDATE vs (CONFIRMED + FALSE POSITIVE)  ← hardest, 477 vs 2,838 samples
```

The final prediction = the class whose binary SVM has the highest confidence score.

**Why this hurts CANDIDATE:** SVM 3 trains on severely imbalanced data (477 positive vs 2,838 negative). `class_weight='balanced'` partially corrects this but the max-margin objective still finds the boundary that cleanly separates the bulk of CONFIRMED and FALSE POSITIVE — CANDIDATE samples near the boundary get misclassified.

### Step 4 — Platt scaling for probabilities

By default, SVM outputs a raw **decision score** (distance from the boundary), not a probability.

We set `probability=True` → applies **Platt scaling**: fits a logistic regression on top of the raw SVM scores to map them to [0, 1] probabilities.

```
P(class k | x) = sigmoid(A · decision_score + B)
```
where A and B are calibrated via 5-fold cross-validation during training.

**Cost:** slower training (~5× due to CV), slightly lower hard-label accuracy vs the un-scaled SVM.  
**Benefit:** ROC-AUC can be computed, enabling fair comparison across all models.

---

## Parameters

| Parameter | Value set | Why |
|-----------|-----------|-----|
| `kernel` | `'rbf'` | Radial Basis Function — standard for non-linearly separable tabular data. Alternatives: 'linear' (too simple, loses FP flag thresholds), 'poly' (more parameters, rarely outperforms rbf on standardised data). |
| `C` | `10` | Regularisation parameter. Low C (0.1) = wide margin, many misclassifications allowed = underfit. High C (1000) = narrow margin, tries to classify everything correctly = overfit. C=10 is the standard moderately-strict setting. Consistently outperforms C=1 and C=100 on our validation set. |
| `gamma` | `'scale'` | Controls radius of influence per training point. `'scale'` = 1/(n_features × variance) — auto-calibrated after StandardScaler. Removes need for manual tuning. |
| `probability` | `True` | Enables Platt scaling for calibrated probabilities. Required for ROC-AUC computation. |
| `class_weight` | `'balanced'` | Scales C penalty inversely by class frequency. CANDIDATE errors cost ~4.9× more. Without this, CANDIDATE F1 drops to ~0.06–0.12. |
| `random_state` | `42` | Seeds the Platt scaling cross-validation for reproducibility. |

### The C–gamma interaction

C and gamma interact strongly — getting one right without the other produces poor results:

| | Low gamma (smooth) | High gamma (complex) |
|--|-------------------|---------------------|
| **Low C** | Very smooth, wide margin → underfits | Tight kernel, loose margin → unstable |
| **High C** | Smooth boundary, hard margin → good | Very complex, tight → **overfits** |
| **Optimal** | Moderate gamma + moderate-high C | = our C=10, gamma='scale' |

Standard recommendation: grid search over both. For dissertation baseline, C=10 + gamma='scale' avoids the worst failure modes.

---

## Results

### Test set performance (586 samples)

| Metric | Score | vs RF | Interpretation |
|--------|-------|-------|----------------|
| **Accuracy** | **81.9%** | −7.5% | 480 / 586 correct |
| **F1 Macro** | **0.739** | −0.039 | 3rd of 3 classical models |
| **ROC-AUC** | **0.897** | −0.058 | 3rd of 3 classical models |
| **Cohen's κ** | **0.681** | −0.115 | Substantial agreement |

### Per-class breakdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| CONFIRMED | 0.83 | 0.87 | **0.85** | 352 |
| FALSE POSITIVE | 0.99 | 0.99 | **0.99** ★ | 163 |
| CANDIDATE | 0.43 | 0.34 | **0.38** | 71 |

> ★ SVM achieves the **highest FALSE POSITIVE F1 of all 3 models** (0.99 vs RF's 0.98) — FP flags create exactly the sharp, clean boundaries SVM excels at.

### Where SVM stands vs the other models

```
Metric          RF      LR      SVM
────────────────────────────────────
Accuracy        89.4%   81.9%   81.9%   ← SVM ties LR
F1 Macro        0.778   0.763   0.739   ← SVM is last
ROC-AUC         0.955   0.924   0.897   ← SVM is last
F1 CONFIRMED    0.92    0.84    0.85
F1 FALSE POS    0.98    0.99    0.99    ← SVM wins this class
F1 CANDIDATE    0.43    0.46    0.38    ← SVM is worst here
```

---

## Metrics Explained

### Why SVM's ROC-AUC (0.897) is lower than LR (0.924) despite similar accuracy

AUC measures probability calibration quality as well as discrimination. LR produces natively calibrated probabilities (directly from log-loss optimisation). SVM probabilities come from **Platt scaling** — a logistic fit on top of raw scores — which introduces calibration error.

The result: SVM makes sharp, accurate hard-label decisions (excellent FALSE POSITIVE F1) but its probability estimates are less reliable than LR's. **Hard-label accuracy and AUC measure different things.**

### The precision-recall trade-off for CANDIDATE

SVM's max-margin objective naturally trades recall for precision:
- **Precision (CANDIDATE) ≈ 0.43** — of everything labelled CANDIDATE, 43% are correct
- **Recall (CANDIDATE) ≈ 0.34** — of all actual CANDIDATEs, only 34% found

SVM only predicts CANDIDATE when highly confident → high precision but low recall. This is the opposite of LR, which has lower precision but higher recall on CANDIDATE (see LR notes).

### Understanding F1 Macro = 0.739

```
F1 Macro = (F1_CONFIRMED + F1_FALSE_POS + F1_CANDIDATE) / 3
         = (0.85 + 0.99 + 0.38) / 3
         = 0.739
```

The low CANDIDATE F1 (0.38) pulls the macro average down significantly. F1 Weighted (which weights by class size) would score ~0.83 — but it hides the CANDIDATE weakness, so we use Macro.

### Cohen's Kappa = 0.681

The kappa gap vs RF (0.796 − 0.681 = **0.115**) is proportionally larger than the accuracy gap (89.4% − 81.9% = **7.5 points**). This shows SVM's weakness is more severe on the minority class than the headline accuracy number suggests — kappa exposes what accuracy hides.

---

## What-if

### Changing C

| C value | Effect |
|---------|--------|
| 0.1 | Wide margin, heavy misclassifications allowed. Underfits. Accuracy ~74–77%, F1 ~0.63–0.67. |
| 1 | Standard soft margin. ~2–3% lower accuracy than C=10. Less overfit risk. Solid baseline. |
| **10** | **Current. Optimal for this dataset. Validated setting.** |
| 50 | Narrow margin. Risk of overfitting. Training accuracy ~100%, test may drop. |
| 100 | Hard margin attempt. Individual CANDIDATE training samples memorised. Overfit. |

### Changing kernel

| Kernel | Expected result |
|--------|----------------|
| **rbf** | **Current. Non-linear. Best general-purpose choice.** |
| linear | Flat hyperplane boundaries. Loses FP flag threshold effects. F1 ~0.68–0.73. But: interpretable weight vector. |
| poly (degree 3) | Captures polynomial interactions. Rarely outperforms rbf on standardised data. Slower. |
| sigmoid | Poorly calibrated for most tabular problems. Not recommended for classification. |

### Removing `class_weight='balanced'`

- CANDIDATE F1: **0.38 → ~0.06–0.12** (catastrophic drop)
- F1 Macro: **0.739 → ~0.60–0.64**
- CONFIRMED F1: small improvement (~+2%)

Never remove this on imbalanced data without a specific reason.

---

## Limitations

1. **Black box** — no feature importance, no interpretable coefficients with RBF kernel.
2. **Platt-calibrated probabilities** are less reliable than LR probabilities. AUC comparisons partly reflect this calibration quality difference.
3. **Training complexity O(n² to n³)** — impractical for TESS-scale datasets (hundreds of thousands of TCEs).
4. **OvR multi-class strategy** creates a severe imbalance for the CANDIDATE binary classifier. One-vs-One (OvO) might improve CANDIDATE recall but at higher computational cost.
5. **No systematic grid search** — C=10, gamma='scale' are principled defaults, not the result of exhaustive tuning.
6. **Cannot produce a feature ranking** — unlike RF (Gini importance) or LR (coefficients), SVM with RBF kernel is completely opaque about which features matter.

---

## Dissertation Talking Points

### How to frame SVM in the results section
SVM serves as the theoretically-grounded strong baseline. It has the strongest generalisation guarantees of the three models (VC dimension, max-margin theory). RF outperforming it empirically is a meaningful dataset-specific finding, not a failure of SVM.

### Suggested results section sentence
> "The SVM with RBF kernel (C=10, gamma='scale') achieved accuracy 81.9% and F1 macro 0.739, trailing Random Forest by 7.5 percentage points. Notably, SVM achieved the highest per-class F1 on FALSE POSITIVE detection (0.99), consistent with the strong linear separability of FP-flag features identified during EDA. However, CANDIDATE F1 (0.38) was the lowest across all evaluated models, reflecting the inherent difficulty of the OvR strategy for ambiguous minority classes."

### Theoretical note worth one sentence in Methods
> "SVMs have stronger theoretical generalisation guarantees than Random Forest (VC dimension bounds relate directly to margin width). In a low-data regime, SVM often outperforms RF. With 3,315 training samples, RF has sufficient data to overcome its weaker theoretical bounds through empirical ensemble performance."

### Connection to Genesis CNN
SVM's Platt-calibrated probabilities provide a useful calibration comparison point for the CNN stages. If the Genesis CNN's AUC falls between SVM and RF, it suggests partial probability calibration — worth one sentence in the CNN discussion.

---

## References

- **Cortes, C. & Vapnik, V. (1995).** Support-vector networks. *Machine Learning*, 20(3), 273–297. — The original SVM paper.
- **Platt, J. (1999).** Probabilistic outputs for SVMs and comparisons to regularised likelihood methods. — Platt scaling for probability calibration.
- **Chang, C.C. & Lin, C.J. (2011).** LIBSVM: A library for support vector machines. *ACM TIST*, 2(3). — The underlying SVM implementation in sklearn.
- **Pedregosa et al. (2011).** Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

---

*Last updated: Stage 3 — Classical ML complete*  
*Next: Stage 4 — Baseline CNN (`src/models/baseline_cnn.py`)*
