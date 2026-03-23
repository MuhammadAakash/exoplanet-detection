# ML Evaluation Metrics — Complete Reference Notes
> Exoplanet Candidate Vetting · MSc Data Science Dissertation  
> A standalone reference for all evaluation metrics used across Stages 3–6

---

## Table of Contents
1. [Why metrics matter for imbalanced data](#why-metrics-matter)
2. [The confusion matrix — foundation of everything](#confusion-matrix)
3. [Accuracy](#accuracy)
4. [Precision](#precision)
5. [Recall (Sensitivity)](#recall)
6. [F1 Score](#f1-score)
7. [F1 Macro vs F1 Weighted vs F1 Micro](#f1-variants)
8. [ROC-AUC](#roc-auc)
9. [Cohen's Kappa](#cohens-kappa)
10. [Matthews Correlation Coefficient (MCC)](#mcc)
11. [Which metrics to use when](#which-metrics-to-use)
12. [Our project results — all metrics side by side](#our-results)

---

## Why Metrics Matter

Our dataset is **imbalanced**:
```
Training set:
  CONFIRMED:      2,341 samples  (63.7%)
  FALSE POSITIVE:   860 samples  (23.4%)
  CANDIDATE:        477 samples  (13.0%)
  ─────────────────────────────────────
  Total:          3,678 samples
```

A **naive classifier** that always predicts CONFIRMED would score:
- Accuracy: **63.7%** — looks reasonable!
- F1 Macro: **~0.26** — exposes the uselessness
- ROC-AUC: **0.50** — random chance
- Cohen's Kappa: **0.00** — no better than chance

This is why we never rely on accuracy alone.

---

## Confusion Matrix

The confusion matrix is the **foundation** of all other classification metrics. For our 3-class problem, it is a 3×3 grid:

```
                    PREDICTED
                 CONF    FP    CAND
              ┌────────────────────┐
ACTUAL  CONF  │  TP    FP₂   FP₃  │
        FP    │  FN₂   TP₂   FP₅  │
        CAND  │  FN₃   FN₄   TP₃  │
              └────────────────────┘
```

- **Diagonal** = correct predictions (True Positives for each class)
- **Off-diagonal** = errors (model predicted column class when true class was row class)

### Reading our RF confusion matrix

```
Actual\Predicted   CONFIRMED   FALSE POS   CANDIDATE
CONFIRMED              341          2           9
FALSE POSITIVE           2        161           0
CANDIDATE               40          9          22
```

Key observations:
- 341/352 CONFIRMED correctly identified
- 161/163 FALSE POSITIVE correctly identified (near-perfect)
- 22/71 CANDIDATE correctly identified (hardest class)
- Most CANDIDATE errors → classified as CONFIRMED (40 cases) — makes astrophysical sense

---

## Accuracy

```
Accuracy = (TP₁ + TP₂ + TP₃) / Total
         = (341 + 161 + 22) / 586
         = 524 / 586
         = 89.4%  (for RF)
```

### When accuracy is misleading

Suppose our test set were 590 CONFIRMED and 5 CANDIDATE. Always predicting CONFIRMED:
- Accuracy = 590/595 = **99.2%** — looks amazing
- But: **0/5 CANDIDATE detected** — completely useless

Our test set (60/28/12%) is less extreme but the same principle applies.

### When accuracy is useful

- Quick sanity check
- Communicating results to non-technical audiences
- When classes are roughly balanced (within ~2× of each other)

**Always pair accuracy with F1 Macro and Kappa for technical reporting.**

---

## Precision

```
Precision(class X) = True Positives for X / (True Positives + False Positives for X)
                   = samples correctly labelled X / all samples labelled X
```

**Interpretation:** of everything the model labels as X, what fraction truly is X?

| High precision means | Low precision means |
|---------------------|---------------------|
| Few false alarms | Model is trigger-happy |
| When model says "planet", it's usually right | Model frequently mislabels non-planets as planets |

### Precision for our RF model

```
Precision CONFIRMED    = 341 / (341 + 2 + 40) = 341/383 = 0.890
Precision FALSE POS    = 161 / (161 + 2 + 9)  = 161/172 = 0.936
Precision CANDIDATE    = 22  / (9 + 0 + 22)   = 22/31   = 0.710
```

CANDIDATE precision = 0.710: when RF says CANDIDATE, it is correct 71% of the time. The other 29% are confirmed planets that RF was unsure about.

---

## Recall

```
Recall(class X) = True Positives for X / (True Positives + False Negatives for X)
                = samples correctly labelled X / all actual X samples
```

**Interpretation:** of all actual X candidates in the dataset, what fraction did the model find?

| High recall means | Low recall means |
|-------------------|-----------------|
| Catches most real members | Misses many real members |
| Low false negative rate | High false negative rate |

### Recall for our RF model

```
Recall CONFIRMED    = 341 / 352 = 0.969   (missed 11 confirmed planets)
Recall FALSE POS    = 161 / 163 = 0.988   (missed 2 false positives)
Recall CANDIDATE    = 22  / 71  = 0.310   (missed 49 candidates)
```

CANDIDATE recall = 0.310: RF only finds 22 of the 71 actual candidates. The remaining 49 are mostly labelled as CONFIRMED.

### The precision-recall trade-off

You can always increase recall by predicting a class more aggressively (lowering the classification threshold). But this reduces precision — more false alarms.

```
Example: Predict CONFIRMED for everything
  Recall CONFIRMED    = 1.00  (perfect — never miss a confirmed planet)
  Precision CONFIRMED = 352/586 = 0.60  (but 40% of predictions are wrong)
```

F1 forces a balance between the two.

---

## F1 Score

```
F1(class X) = 2 × Precision × Recall / (Precision + Recall)
            = 2 × TP / (2×TP + FP + FN)
```

The **harmonic mean** of precision and recall. It is low if **either** is low.

Harmonic vs arithmetic mean:
- Arithmetic mean of (0.9, 0.1) = 0.50 — looks okay
- Harmonic mean (F1) of (0.9, 0.1) = 0.18 — exposes the weakness

**F1 rewards balance.** A model with precision=0.5, recall=0.5 has F1=0.50. A model with precision=0.9, recall=0.2 also has F1=0.32 — lower despite higher precision.

### F1 for our models

| Model | F1 CONFIRMED | F1 FALSE POS | F1 CANDIDATE |
|-------|-------------|-------------|-------------|
| RF | **0.92** | 0.98 | 0.43 |
| SVM | 0.85 | **0.99** | 0.38 |
| LR | 0.84 | **0.99** | **0.46** |

---

## F1 Variants

### F1 Macro (what we report as primary metric)

```
F1 Macro = (F1_CONFIRMED + F1_FALSE_POS + F1_CANDIDATE) / 3
```

Simple average across classes. **Treats all three classes equally**, regardless of how many samples each has.

```
RF:  (0.92 + 0.98 + 0.43) / 3 = 0.778
LR:  (0.84 + 0.99 + 0.46) / 3 = 0.763
SVM: (0.85 + 0.99 + 0.38) / 3 = 0.739
```

The poor CANDIDATE F1 (0.38–0.46) pulls macro down significantly — which is **honest**. The model really does struggle with CANDIDATE.

### F1 Weighted

```
F1 Weighted = Σ (class_size / total_samples) × F1_class
```

Weights each class F1 by how many test samples that class has.

```
RF F1 Weighted = (352/586)×0.92 + (163/586)×0.98 + (71/586)×0.43
              = 0.601×0.92 + 0.278×0.98 + 0.121×0.43
              = 0.553 + 0.272 + 0.052
              = 0.877
```

F1 Weighted (0.877) is much higher than F1 Macro (0.778) because it downweights the minority CANDIDATE class (only 12.1% of test samples).

**Why we prefer Macro:** F1 Weighted hides poor minority class performance. For a classification problem where all three classes matter (confirmed planets, FPs, and candidates all have real-world importance), Macro is more honest.

### F1 Micro

```
F1 Micro = 2 × Σ TP_k / (2 × Σ TP_k + Σ FP_k + Σ FN_k)
```

Pools all predictions across classes before computing a single F1. For multi-class with balanced weighting, F1 Micro equals Accuracy. Not commonly reported for multi-class problems.

### Summary: when to use each

| Metric | Use when |
|--------|----------|
| **F1 Macro** | Classes are equally important, dataset is imbalanced → **our choice** |
| **F1 Weighted** | Class importance proportional to frequency (rare) |
| **F1 Micro** | Equivalent to accuracy for multi-class; rarely used |
| **Per-class F1** | Always report alongside aggregate to show class-specific performance |

---

## ROC-AUC

### The ROC Curve

ROC = **Receiver Operating Characteristic**. Plots:
- **Y-axis:** True Positive Rate (= Recall) = TP / (TP + FN)
- **X-axis:** False Positive Rate = FP / (FP + TN)

As you lower the classification threshold from 1.0 → 0.0:
- More samples get predicted as positive
- TPR increases (you catch more real positives)
- FPR also increases (more false alarms)
- The curve traces from (0,0) to (1,1)

```
Perfect classifier:  curve goes through (0,0) → (0,1) → (1,1)  AUC = 1.0
Random classifier:   diagonal line (0,0) → (1,1)                AUC = 0.5
Useful classifier:   curve above the diagonal                   AUC > 0.5
```

### AUC — Area Under the Curve

AUC summarises the entire ROC curve as a single number.

**Probabilistic interpretation:** AUC = probability that the model ranks a randomly chosen positive sample higher than a randomly chosen negative sample.

```
RF AUC = 0.955:  Pick one CONFIRMED and one non-CONFIRMED at random.
                 RF assigns higher P(CONFIRMED) to the actual confirmed one
                 95.5% of the time.
```

### Multi-class AUC — One-vs-Rest macro

For 3 classes, we compute AUC three times:
```
AUC_CONFIRMED     = AUC treating CONFIRMED as positive, rest as negative
AUC_FALSE_POS     = AUC treating FALSE POS as positive, rest as negative
AUC_CANDIDATE     = AUC treating CANDIDATE as positive, rest as negative

AUC_macro = (AUC_CONFIRMED + AUC_FALSE_POS + AUC_CANDIDATE) / 3
```

### Why AUC matters for this project specifically

In a real telescope scheduling system, you would:
1. Compute P(CONFIRMED) for all ~3,900 KOIs
2. Sort by probability
3. Schedule the top-ranked candidates for follow-up

AUC directly measures the quality of this ranking. RF's AUC=0.955 means its ranking is excellent for prioritising telescope time.

### AUC is threshold-independent

Unlike F1, accuracy, and Kappa — which all depend on the specific 50% threshold — AUC evaluates the model across **all possible thresholds simultaneously**. This makes it more robust and useful when the optimal threshold is unknown.

### Our model AUC scores

| Model | ROC-AUC | Interpretation |
|-------|---------|----------------|
| **RF** | **0.955** | Best ranking quality |
| LR | 0.924 | Well-calibrated probabilities, second-best ranking |
| SVM | 0.897 | Platt scaling limits calibration quality |

---

## Cohen's Kappa

### The problem Kappa solves

A model that randomly predicts classes with the training set proportions (63.7% CONFIRMED, 23.4% FP, 12.9% CAND) would score:

```
Expected accuracy = 0.637² + 0.234² + 0.129²
                  = 0.406 + 0.055 + 0.017
                  = 0.477  (47.7%)
```

A model scoring 89.4% accuracy is not 89.4% good — it is only (89.4% − 47.7%) = 41.7% better than chance. But chance itself was 47.7%, so the real improvement above chance is 41.7%/(100% − 47.7%) = **79.6%**. That is Cohen's Kappa.

### Formula

```
κ = (Observed accuracy − Expected accuracy) / (1 − Expected accuracy)
```

For RF:
```
κ = (0.894 − 0.477) / (1 − 0.477)
  = 0.417 / 0.523
  = 0.796
```

### Interpretation scale

```
κ < 0.00   = Worse than chance (model is anti-correlated with truth)
0.00–0.20  = Slight agreement
0.21–0.40  = Fair agreement
0.41–0.60  = Moderate agreement
0.61–0.80  = Substantial agreement  ← LR (0.692) and SVM (0.681) are here
0.81–1.00  = Near-perfect agreement ← RF (0.796) approaches this
```

### Why Kappa is expected in scientific papers

Kappa corrects for class imbalance that inflates raw accuracy. It is the standard agreement metric in:
- Medical diagnosis papers (where class imbalance is common)
- Remote sensing classification (land cover, multi-class)
- Any scientific classification study with imbalanced classes

Reporting Kappa signals methodological rigour to dissertation examiners.

### Our Kappa scores

| Model | Kappa | Interpretation |
|-------|-------|----------------|
| **RF** | **0.796** | Near top of "substantial" |
| LR | 0.692 | Solid "substantial" |
| SVM | 0.681 | Solid "substantial" |

Note: the Kappa gap RF vs SVM (0.796 − 0.681 = 0.115) is proportionally larger than the accuracy gap (89.4% − 81.9% = 7.5 points). Kappa exposes the minority class weakness that accuracy hides.

---

## Matthews Correlation Coefficient (MCC)

```
MCC = (TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

MCC is similar to Kappa but based on the product-moment correlation between actual and predicted labels. Range: −1 (perfectly wrong) to +1 (perfectly correct), 0 = random.

For multi-class:
```
MCC = (Σ_k Σ_l Σ_m C_{kk}C_{ml} - C_{lk}C_{km}) / √(...)
```

MCC is considered **the most informative single metric** for binary imbalanced classification by many researchers. For multi-class, it is less commonly reported than F1 Macro and Kappa but increasingly seen in NeurIPS and ICML papers.

We compute and save MCC in `evaluation/metrics.py` but lead with F1 Macro and Kappa for the dissertation.

---

## Which Metrics to Use When

### For this dissertation
```
Primary:   F1 Macro       — honest, imbalance-corrected, equal class weight
Secondary: ROC-AUC        — threshold-independent, measures ranking quality
Secondary: Cohen's Kappa  — standard scientific agreement metric
Reference: Accuracy       — familiar to all readers, easy to communicate
Detail:    Per-class F1   — exposes CANDIDATE weakness explicitly
```

### For different audiences

| Audience | Lead metric | Why |
|----------|-------------|-----|
| Technical (dissertation examiners) | F1 Macro + Kappa | Scientifically rigorous, imbalance-aware |
| Telescope scheduling team | ROC-AUC | They need ranking quality, not just labels |
| General public | Accuracy | Most familiar and understandable |
| NASA Robovetter comparison | F1 per class | Direct comparison of each class performance |

### Common mistakes to avoid

| Mistake | Why it is wrong | What to do instead |
|---------|-----------------|-------------------|
| Reporting only accuracy | Misleading for imbalanced data | Always add F1 Macro and Kappa |
| Reporting F1 Weighted as "F1" | Hides minority class weakness | Specify Macro or Weighted explicitly |
| Ignoring per-class breakdown | Hides that CANDIDATE is weak | Always include per-class F1 table |
| Comparing AUC without checking calibration | SVM Platt vs LR log-loss | Note calibration method alongside AUC |
| Calling 0.43 CANDIDATE F1 "poor model performance" | It is a data problem | Explain the CANDIDATE class definition |

---

## Our Results — All Metrics Side by Side

### Test set: 586 samples (352 CONFIRMED · 163 FALSE POSITIVE · 71 CANDIDATE)

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC | Cohen's κ |
|-------|----------|----------|-------------|---------|-----------|
| **RF** | **89.4%** | **0.778** | 0.877 | **0.955** | **0.796** |
| LR | 81.9% | 0.763 | 0.862 | 0.924 | 0.692 |
| SVM | 81.9% | 0.739 | 0.831 | 0.897 | 0.681 |

### Per-class F1

| Model | F1 CONFIRMED | F1 FALSE POS | F1 CANDIDATE |
|-------|-------------|-------------|-------------|
| **RF** | **0.92** | 0.98 | 0.43 |
| SVM | 0.85 | **0.99** | 0.38 |
| LR | 0.84 | **0.99** | **0.46** |

### Summary of metric "winners"

```
Best Accuracy:      Random Forest (89.4%)
Best F1 Macro:      Random Forest (0.778)
Best ROC-AUC:       Random Forest (0.955)
Best Cohen's κ:     Random Forest (0.796)
Best F1 CONFIRMED:  Random Forest (0.92)
Best F1 FALSE POS:  SVM and LR tied (0.99)
Best F1 CANDIDATE:  Logistic Regression (0.46)  ← surprise result
```

The pattern common to all three models: FALSE POSITIVE is easy (F1 > 0.98) and CANDIDATE is hard (F1 < 0.46). This is not a modelling failure — it is a property of the data.

---

*Last updated: Stage 3 — Classical ML complete*  
*These metrics are reused for Stages 4 (Baseline CNN) and 5 (Genesis CNN) — same evaluation code in `src/evaluation/metrics.py`*
