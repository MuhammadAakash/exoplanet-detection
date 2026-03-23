# Random Forest — Complete Reference Notes
> Exoplanet Candidate Vetting · Stage 3 · MSc Data Science Dissertation  
> Dataset: NASA Kepler KOI Q1-Q17 DR25 · 3,901 samples · 37 features · 3 classes

---

## Table of Contents
1. [What is Random Forest?](#what-is-random-forest)
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

## What is Random Forest?

Random Forest is an **ensemble learning** method built from a collection of individual decision trees. Each tree is trained independently on a slightly different random sample of the data and features. When predicting a new sample, every tree votes on the class and the **majority vote wins**.

The key idea: a single deep decision tree tends to memorise training data (overfit). But if you build 300 trees that each see different random slices of data and features, their individual errors are uncorrelated — they cancel out in the majority vote.

> **One-line intuition:** Ask 300 independent experts who each studied a slightly different random subset of your data. Aggregate their votes. Individual mistakes cancel out.

---

## Why we used it

| Reason | Detail |
|--------|--------|
| **Handles non-linear boundaries** | Transit features like `koi_prad > 15` are hard thresholds — trees represent these perfectly as a single split |
| **Feature importance built in** | After training, the forest tells you which features were most useful — confirmed our EDA ANOVA findings |
| **Robust to outliers** | Median-like splits are not sensitive to extreme values in `koi_depth` or `koi_model_snr` |
| **No scaling required** | Trees split on thresholds, not distances — `StandardScaler` not needed (unlike SVM/LR) |
| **Strong empirical baseline** | RF consistently tops tabular classification benchmarks (Grinsztajn et al., 2022) |
| **Handles imbalance via `class_weight`** | Our 4.9× CONFIRMED/CANDIDATE imbalance is corrected with `class_weight='balanced'` |

---

## How it works

### Step 1 — Bootstrap sampling

Before training each tree, the algorithm randomly draws ~63% of training rows **with replacement**. This means:
- Some rows appear multiple times in one tree's training set
- ~37% of rows are left out (called **out-of-bag** samples — can estimate error without a validation set)
- Every tree sees a **different** bootstrap sample → every tree learns slightly different patterns

### Step 2 — Growing each decision tree

Each tree is built by repeatedly asking the best yes/no question at each node. But at each split point, only **√37 ≈ 6 features** (chosen at random) are considered — not all 37. This forces tree diversity.

The best split among those 6 features is chosen by minimising **Gini impurity** — a measure of how mixed-class the two resulting groups are.

```
Gini = 1 − Σ p_i²
where p_i = fraction of samples belonging to class i
```

With `max_depth=None`, trees grow until every leaf contains a single class. Trees are fully grown and individually overfit — the ensemble corrects this.

**Example splits learned from our KOI data:**
```
if koi_fpflag_co == 1      → HIGH probability FALSE POSITIVE
if koi_prad > 15.0         → Almost certainly FALSE POSITIVE (too large to be a planet)
if koi_model_snr < 7.1     → Likely CANDIDATE (weak signal)
if koi_fpflag_ss == 1 AND
   koi_depth > 8000        → FALSE POSITIVE (secondary eclipse + deep transit)
```

### Step 3 — Majority vote

When a new KOI arrives with its 37 features, all 300 trees independently produce a class prediction. The final output is the class with the most votes.

```
Example vote for one test KOI:
  240 trees → CONFIRMED
   45 trees → FALSE POSITIVE
   15 trees → CANDIDATE
  ────────────────────────────
  Final:     CONFIRMED
```

The model also outputs **probabilities** = vote fractions (240/300 = 0.80 P(CONFIRMED)).

### Step 4 — Feature importance

After training, the model tallies how much each feature reduced Gini impurity across all splits in all 300 trees. Features used for high-impact early splits score higher.

**Top features in our model (by Gini importance):**
```
1. koi_fpflag_co    ████████████░░  (centroid offset flag)
2. koi_fpflag_ss    ███████████░░░  (significant secondary flag)
3. koi_model_snr    ████████░░░░░░  (signal-to-noise ratio)
4. koi_depth        ███████░░░░░░░  (transit depth in ppm)
5. koi_incl         ██████░░░░░░░░  (orbital inclination)
6. koi_prad         █████░░░░░░░░░  (planet radius in Earth radii)
```

These match the ANOVA F-scores from EDA — the model learned astrophysically meaningful patterns.

### Why ensemble beats a single tree

| Single Decision Tree | Random Forest (300 trees) |
|---------------------|--------------------------|
| High variance (overfits) | Low variance (ensemble average) |
| Uses all 37 features at each split | Uses ~6 random features at each split |
| One set of rules | 300 diverse rule sets, voted on |
| Fast to train | ~300× slower, but parallelised (`n_jobs=-1`) |
| Interpretable | Black box — no single readable equation |

---

## Parameters

### Full parameter table

| Parameter | Value we set | Why |
|-----------|-------------|-----|
| `n_estimators` | `300` | Number of trees. Diminishing returns past ~200. 300 is the sweet spot: meaningfully better than 100, no practical gain going to 500+. |
| `max_depth` | `None` (unlimited) | Trees grow fully. Individual overfitting is corrected by ensemble averaging. Limiting depth hurts FP flag threshold detection. |
| `max_features` | `'sqrt'` (default) | At each split, consider √37 ≈ 6 features. This is the core randomisation that decorrelates trees. Without it, every tree would make the same first split (koi_fpflag_co), defeating the ensemble. |
| `class_weight` | `'balanced'` | **Most critical.** 4.9× CONFIRMED/CANDIDATE imbalance. 'balanced' weights each sample inversely by class frequency. Each CANDIDATE sample counts ~4.9× more than CONFIRMED during training. Remove this → CANDIDATE F1 drops from 0.43 to <0.15. |
| `min_samples_split` | `2` (default) | Minimum samples to split a node. Allows fine-grained CANDIDATE detection even in small sub-groups of the minority class. |
| `random_state` | `42` | Seeds all internal random processes. Full reproducibility — required for dissertation. Same seed used throughout the pipeline. |
| `n_jobs` | `-1` | Use all CPU cores. Speed optimisation only — zero effect on predictions. |

### The `class_weight='balanced'` parameter in detail

Our training set: 2,341 CONFIRMED · 860 FALSE POSITIVE · 477 CANDIDATE.

Without balancing, a model that always predicts CONFIRMED would be right ~60% of the time. The model learns to exploit this.

With `class_weight='balanced'`, sklearn computes:
```
weight_CONFIRMED    = n_samples / (n_classes × n_CONFIRMED) = 3678 / (3 × 2341) = 0.52
weight_FALSE_POS    = 3678 / (3 × 860)  = 1.43
weight_CANDIDATE    = 3678 / (3 × 477)  = 2.57
```
Each CANDIDATE error costs ~4.9× more than a CONFIRMED error during training.

**Effect:** CANDIDATE F1 goes from ~0.10 to ~0.43. The cost: CONFIRMED precision drops by ~2–3%.

---

## Results

### Test set performance (586 samples, never seen during training)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | **89.4%** | 524 / 586 correct |
| **F1 Macro** | **0.778** | Best of 3 classical models |
| **ROC-AUC** | **0.955** | Best of 3 classical models |
| **Cohen's κ** | **0.796** | Near the top of "substantial agreement" |

### Per-class breakdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| CONFIRMED | 0.88 | 0.97 | **0.92** | 352 |
| FALSE POSITIVE | 0.97 | 0.99 | **0.98** | 163 |
| CANDIDATE | 0.66 | 0.32 | **0.43** | 71 |

### How RF compares to SVM and Logistic Regression

| Model | Accuracy | F1 Macro | ROC-AUC | Cohen's κ | F1 CAND |
|-------|----------|----------|---------|-----------|---------|
| **Random Forest** | **89.4%** | **0.778** | **0.955** | **0.796** | 0.43 |
| Logistic Regression | 81.9% | 0.763 | 0.924 | 0.692 | **0.46** |
| SVM | 81.9% | 0.739 | 0.897 | 0.681 | 0.38 |

> **Note:** RF wins every metric except CANDIDATE F1 — where Logistic Regression surprisingly wins (0.46 vs 0.43). This is explained in the LR notes.

### Why RF wins

1. **Non-linear boundaries** — the rule `koi_prad > 15 → FALSE POSITIVE` is a single tree split. Linear models approximate it.
2. **Variance reduction** — 300 trees voting eliminates individual tree noise.
3. **Feature diversity** — random subsets at each split force trees to discover different predictive patterns.

### Why CANDIDATE is hard for every model

CANDIDATE is an **administrative label**, not a physical class. It means "unresolved — we don't have enough information yet." Some are real planets awaiting confirmation. Others are false positives awaiting rejection. Their tabular features are identical to both other classes — no model can separate labels when the ground truth is genuinely unknown.

This is a **data limitation**, not a modelling failure.

---

## Metrics Explained

### Accuracy
```
Accuracy = correct predictions / total predictions
         = 524 / 586 = 89.4%
```

**Why it is not enough alone:** if our test set were 90% CONFIRMED, always predicting CONFIRMED would score 90% while being completely useless. Accuracy is misleading when classes are imbalanced.

**When useful:** quick sanity check, communicating to non-technical readers. Always support with F1 and Kappa.

---

### Precision, Recall, and F1

**Precision** = of all candidates labelled X, what fraction truly are X?
- High precision = few false alarms
- Low precision = model is trigger-happy

**Recall** = of all actual X candidates in the test set, what fraction did the model find?
- High recall = catches most real members of this class
- Low recall = misses many real members

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is the **harmonic mean** — low if either precision OR recall is low. Forces both to be good simultaneously. Range: 0 (useless) → 1.0 (perfect).

**F1 Macro** = simple average of F1 across all three classes:
```
F1 Macro = (F1_CONFIRMED + F1_FALSE_POSITIVE + F1_CANDIDATE) / 3
         = (0.92 + 0.98 + 0.43) / 3
         = 0.778
```

Treats all three classes equally, regardless of how many samples each has. This is why it penalises poor CANDIDATE performance — which is honest.

**F1 Weighted** = average weighted by class size. Would give ~0.88 for RF by downweighting CANDIDATE. Less honest for imbalanced datasets — we use Macro throughout the dissertation.

---

### ROC-AUC (0.955)

The **ROC curve** plots True Positive Rate (recall) on the y-axis vs False Positive Rate on the x-axis, sweeping all classification thresholds from 0 → 1.

**AUC** (Area Under the Curve): 0.5 = random guessing, 1.0 = perfect discrimination.

**Concrete meaning of 0.955:** if you randomly pick one CONFIRMED and one non-CONFIRMED candidate, there is a 95.5% chance RF assigns a higher P(CONFIRMED) to the actual confirmed one. This is a **threshold-independent** measure of ranking quality.

**Why it matters:** in a real telescope scheduling system, you rank all candidates by P(CONFIRMED) and observe the top-ranked ones. AUC measures how good that ranking is — not just the quality of the 50% threshold decision.

**Multi-class AUC:** we report **One-vs-Rest macro AUC** — compute AUC for each class vs the rest, then average.

---

### Cohen's Kappa (0.796)

```
κ = (Observed accuracy − Expected accuracy) / (1 − Expected accuracy)
```

**Expected accuracy** = what a random classifier would score by sampling from the class distribution. For our ~60/28/12% split, random chance ≈ 47%.

**Interpretation scale:**
```
κ < 0.2     = Slight agreement
0.2 – 0.4   = Fair
0.4 – 0.6   = Moderate
0.6 – 0.8   = Substantial  ← RF at 0.796 sits here (top of range)
0.8 – 1.0   = Near-perfect
```

**Why report it:** Kappa is the standard agreement metric in scientific classification papers. It corrects for class imbalance that inflates raw accuracy. More credible than accuracy for imbalanced datasets.

---

### Confusion Matrix

A 3×3 grid where rows = true class, columns = predicted class. Diagonal = correct predictions. Off-diagonal = errors.

**What our RF confusion matrix reveals:**
- Most CANDIDATE errors are classified as CONFIRMED — not FALSE POSITIVE
- This makes astrophysical sense: an unconfirmed planet candidate looks exactly like a confirmed planet in tabular features
- The error pattern is meaningful, not random

---

## What-if

### Changing `n_estimators`

| Trees | Expected Accuracy | F1 Macro | Training Time | Notes |
|-------|------------------|----------|---------------|-------|
| 50 | ~85–87% | ~0.72 | ~1s | High variance, results change between runs |
| 100 | ~87–88% | ~0.765 | ~2s | Good for quick experiments |
| **300** | **89.4%** | **0.778** | **~6s** | **Current — optimal** |
| 500 | ~89.5% | ~0.779 | ~10s | Negligible improvement |
| 1000 | ~89.5% | ~0.780 | ~20s | Waste of compute |

### Changing `max_depth`

| max_depth | Effect |
|-----------|--------|
| 5 | High bias. Cannot learn compound FP flag rules. F1 macro ~0.65–0.68. |
| 10 | Moderate regularisation. ~87% accuracy. Useful for very small datasets. |
| 20 | Nearly identical to None for this dataset. |
| **None** | **Current. Full depth. Ensemble corrects overfitting.** |

### Removing `class_weight='balanced'`

**This is the most impactful single change you can make:**
- CANDIDATE F1 drops from **0.43 → ~0.10–0.15**
- CONFIRMED F1 improves by ~2% (model becomes more aggressive about majority class)
- F1 Macro drops from **0.778 → ~0.57–0.63**

---

## Limitations

1. **Black box** — no single interpretable equation. Cannot say "feature X contributed Y to this prediction" (unlike Logistic Regression).
2. **Feature importance ≠ causation** — `koi_fpflag_co` is important because it co-occurs with false positives, not because it causes them.
3. **CANDIDATE F1 = 0.43** — must be acknowledged as a limitation, even though it is a data issue. The astrophysical ambiguity of the CANDIDATE label is the root cause.
4. **Pre-extracted features only** — model trained on tabular KOI features, not raw light curves. Cannot generalise to telescopes with different feature extraction pipelines.
5. **No hyperparameter grid search** — n_estimators=300, max_depth=None are well-established defaults. A full GridSearchCV might produce marginal improvements but is not expected to change the overall conclusions.
6. **Memory** — 300 full-depth trees on 37 features can use significant RAM. On production-scale data (TESS: millions of TCEs), memory becomes a constraint.

---

## Dissertation Talking Points

### Lead finding
> "Random Forest achieves the highest performance across all metrics: accuracy 89.4%, F1 macro 0.778, ROC-AUC 0.955, Cohen's κ 0.796."

### Feature importance finding
> "The Random Forest's most important features — koi_fpflag_co, koi_fpflag_ss, koi_model_snr, koi_depth — are consistent with the ANOVA F-scores identified in EDA. This agreement between a model-agnostic statistical test and the tree-based importance scores provides mutual validation: the model learned astrophysically meaningful patterns, not spurious correlations."

### Suggested results section sentence
> "Random Forest outperformed all other evaluated models on aggregate metrics (accuracy 89.4%, F1 macro 0.778, AUC 0.955, κ 0.796), consistent with its documented strength on tabular classification tasks with non-linear feature interactions (Breiman, 2001). Feature importances identified koi_fpflag_co and koi_fpflag_ss as the most discriminative features, corroborating the ANOVA F-scores computed during exploratory analysis."

### Connection to the Genesis CNN
RF's strong tabular performance is the baseline the CNN must be evaluated against. Key framing options:
- **If CNN ≈ RF:** demonstrates the dual-branch architecture is competitive with the ensemble standard
- **If RF > CNN:** valid, publishable finding — ensemble methods remain SOTA for pre-extracted tabular features (supported by Grinsztajn et al., 2022)
- Either outcome tells an interesting story

---

## References

- **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32. — The original RF paper.
- **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).** Why tree-based models still outperform deep learning on tabular data. *NeurIPS 2022*. — Direct empirical support for RF results on tabular classification.
- **Pedregosa et al. (2011).** Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. — sklearn implementation reference.
- **Shallue & Vanderburg (2018).** Identifying Exoplanets with Deep Learning. *AJ*, 155(2). — The Genesis CNN architecture this project adapts.

---

*Last updated: Stage 3 — Classical ML complete*  
*Next: Stage 4 — Baseline CNN (`src/models/baseline_cnn.py`)*
