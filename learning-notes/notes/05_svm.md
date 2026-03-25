# 📚 Notes 05 — Support Vector Machine (SVM)

---

## The Core Idea — Drawing the Best Dividing Line

Imagine plotting two types of stars on a graph. You want to draw a line that separates them.

```
      ●  ●  ●  ●
   ●  ●  ●  ●         ← Green dots (planets / confirmed)
      ●  ●
─────────────────────  ← Dividing line
         ○  ○  ○
      ○  ○  ○  ○      ← Red dots (false positives)
         ○  ○
```

Many lines could separate them. SVM asks: **which line gives the maximum breathing room between the two groups?**

---

## The Margin — Maximum Breathing Room

SVM finds the line that is as **far as possible** from both groups simultaneously. This gap is called the **margin**.

```
   ●  ●  ●
●  ●  ●               ← Planets (confirmed)
   ●    ┌─────────────────┐
        │   Maximum Gap   │  ← Margin (breathing room)
        │   (the margin)  │
        └─────────────────┘
      ○  ○             ← False positives
○  ○  ○  ○
```

The points sitting right on the edge of the margin are called **Support Vectors** — they define where the boundary goes. This is where the name "Support Vector Machine" comes from.

### Why Maximise the Margin?

A wider margin = better generalisation to new data.

> **Analogy:** If you're parking between two cars, you want maximum space on both sides. A car squeezed right against one side — any tiny movement could cause a crash. Maximum margin = safety buffer.

A decision boundary with more breathing room is less likely to misclassify slightly unusual new examples.

---

## The Kernel Trick — Handling Complex Data

Real data is almost never separable by a straight line. SVM solves this with the **kernel trick**.

### The Problem

```
2D view (can't separate with a line):

   ○   ●   ○
 ○   ○   ●   ●
   ●   ○   ●
```

### The Solution — Go to Higher Dimensions

The kernel mathematically transforms the data into a higher-dimensional space where a flat surface CAN separate the groups.

> **Analogy:** Red and green marbles mixed on a table (2D). Can't draw a line to separate them. But lift the green ones up (add a 3rd dimension) and now a flat sheet of paper separates them perfectly.

### Your Project's Kernel: RBF (Radial Basis Function)

```python
SVC(kernel="rbf", ...)
```

RBF creates **circular/spherical** decision boundaries instead of straight lines.

It measures: *"how similar is this new point to each training point?"*

Points closer together in feature space get higher similarity scores. RBF uses this to create smooth, curved boundaries that fit complex data much better than a straight line.

---

## The C Parameter — Tolerance for Mistakes

Real data is messy. Some points will always be on the wrong side. The C parameter controls your tolerance.

```
Low C (e.g. C=0.1)        High C (e.g. C=100)
─────────────────          ─────────────────
Wide margin                Narrow margin
Some misclassifications    Tries to classify everything
allowed                    correctly
Better generalisation      Risk of overfitting
```

| C Value | Effect | Use When |
|---|---|---|
| Very low (0.01–0.1) | Very wide margin, many errors tolerated | Very noisy data |
| Medium (1–10) | Balance between margin and accuracy | Most situations |
| Very high (100+) | Tries to get everything right | Clean, noise-free data |

**In your project: `C=10`** — moderately strict. Fits the data well without squeezing the margin too much.

---

## The Gamma Parameter (for RBF)

Gamma controls **how far the influence of each training example reaches**.

```
Low gamma:           High gamma:
─────────────        ─────────────
Smooth, wide         Jagged, tight
decision boundary    decision boundary
Each point has       Each point only
far-reaching         influences nearby
influence            points
```

**In your project: `gamma="scale"`** — automatically set to `1 / (n_features × X.var())`. This scales well with your 37 features.

---

## Why SVM Needs StandardScaler

SVM measures **distances** between points in feature space. If features have wildly different scales:

```
Without scaling:
  koi_depth  → values: 50 to 500,000 ppm    ← huge range
  koi_impact → values: 0.0 to 1.0           ← tiny range

SVM sees koi_depth as ~100,000× more important than koi_impact
because the distances are dominated by the large-scale feature.
```

```
After StandardScaler:
  koi_depth  → values: -2.0 to +3.5  ← normalised
  koi_impact → values: -1.8 to +2.1  ← normalised

Both features now contribute equally to distance calculations.
```

> ⚠️ **Critical:** SVM without StandardScaler gives misleading results. Always scale first.

This is why your preprocessing pipeline applies StandardScaler before training SVM (and all other models that need it).

---

## SVM for Multi-Class Problems

SVM was originally designed for binary classification (two classes). For your 3-class problem, scikit-learn uses **One-vs-Rest (OvR)**:

```
Train 3 separate binary classifiers:
  Classifier 1: CONFIRMED vs (FALSE POSITIVE + CANDIDATE)
  Classifier 2: FALSE POSITIVE vs (CONFIRMED + CANDIDATE)
  Classifier 3: CANDIDATE vs (CONFIRMED + FALSE POSITIVE)

For new star:
  Classifier 1 says: 0.82 confidence of CONFIRMED
  Classifier 2 says: 0.31 confidence of FALSE POSITIVE
  Classifier 3 says: 0.19 confidence of CANDIDATE

Final prediction: CONFIRMED (highest confidence)
```

---

## Your SVM Configuration

```python
SVC(
    kernel="rbf",           # RBF kernel — curved boundaries
    C=10,                   # Moderately strict margin
    gamma="scale",          # Auto-scaled gamma
    probability=True,       # Enable probability estimates (for ROC-AUC)
    class_weight="balanced",# Pay more attention to rare classes
    random_state=42,        # Reproducibility
)
```

**Why `probability=True`?** By default SVM only outputs the class prediction. To calculate ROC-AUC, you need probability scores (0-1). Setting `probability=True` enables this via Platt scaling — adds a small computational cost.

---

## Results in Your Project

| Metric | SVM | Comparison to RF |
|---|---|---|
| Accuracy | 86.1% | −3.3% below Random Forest |
| F1 Macro | 0.783 | −0.038 below RF |
| ROC-AUC | 0.944 | −0.018 below RF |
| Cohen's κ | 0.783 | −0.045 below RF |

**Why does SVM underperform RF here?**
- SVM with RBF creates smooth continuous boundaries — tabular astronomical data has complex, irregular patterns that trees handle better
- Random Forest's feature importance naturally emphasises the binary FP flags; SVM treats all features as continuous distances

---

## SVM vs Random Forest — Head to Head

| Aspect | SVM | Random Forest |
|---|---|---|
| Core mechanism | Maximum margin boundary | Majority vote of 300 trees |
| Needs feature scaling? | ✅ YES — critical | ❌ No |
| Handles missing values? | ❌ No (after imputation) | ✅ Better natively |
| Interpretability | ❌ Black box | ✅ Feature importance |
| Speed on large data | 🐢 Slow (O(n²–n³)) | 🐇 Faster |
| Best for | Clean, well-separated data | Complex, messy tabular data |
| Your project result | 86.1% accuracy | 89.4% accuracy |

---

## Key Concepts Summary

```
Support Vectors
    └── Training points closest to the decision boundary
        — they define where the boundary goes

Margin
    └── The gap between the boundary and the nearest points
        — SVM maximises this gap

Kernel
    └── Mathematical transformation to handle non-linear data
        — RBF: circular boundaries (used in your project)

C Parameter
    └── Controls tolerance for misclassification
        — Low C = wide margin, some errors
        — High C = narrow margin, fewer errors (risk of overfit)

Gamma
    └── How far each training point's influence reaches
        — Low = smooth boundary, High = jagged boundary

One-vs-Rest
    └── How SVM handles 3+ classes
        — Trains one binary classifier per class
```

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **SVM** | Support Vector Machine — finds maximum margin decision boundary |
| **Support Vectors** | Training points closest to the decision boundary |
| **Margin** | The gap between the boundary and nearest points — SVM maximises this |
| **Kernel** | Mathematical trick to create non-linear decision boundaries |
| **RBF Kernel** | Radial Basis Function — creates circular/spherical boundaries |
| **C parameter** | Controls trade-off between margin width and training accuracy |
| **Gamma** | Controls how far influence of each training point reaches |
| **One-vs-Rest** | Strategy for multi-class SVM — one classifier per class |
| **Platt Scaling** | Method for converting SVM outputs to probability scores |
| **StandardScaler** | Transforms features to mean=0, std=1 — essential before SVM |

---

*Notes from MSc Data Science Dissertation — Exoplanet Candidate Vetting*
