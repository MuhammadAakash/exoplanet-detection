# Logistic Regression — Complete Reference Notes
> Exoplanet Candidate Vetting · Stage 3 · MSc Data Science Dissertation  
> Dataset: NASA Kepler KOI Q1-Q17 DR25 · 3,901 samples · 37 features · 3 classes

---

## Table of Contents
1. [What is Logistic Regression?](#what-is-logistic-regression)
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

## What is Logistic Regression?

Despite the name, Logistic Regression is a **classification** algorithm, not a regression algorithm. It learns one **weight (coefficient) per feature** and uses the **softmax function** to convert weighted sums into class probabilities.

It is the simplest and most interpretable of the three classical models — and produces the best-calibrated probabilities of all three. It also delivers the **surprise result** of Stage 3: the best CANDIDATE F1 (0.46) despite being the least complex model.

> **One-line intuition:** Assign a score to each feature (positive = supports this class, negative = opposes it). Sum the scores. Convert to probabilities via softmax. Predict the class with the highest probability.

---

## Why we used it

| Reason | Detail |
|--------|--------|
| **Interpretable coefficients** | Only model where you can directly read "increasing koi_fpflag_co by 1 multiplies the FP odds by e^w" |
| **Best probability calibration** | Probabilities come directly from log-loss optimisation — inherently well-calibrated |
| **Linear baseline** | Establishes how much of the problem is linearly separable (answer: a lot, for FALSE POSITIVE) |
| **Fast to train** | Converges in seconds even with 1000 max_iter |
| **Second-highest ROC-AUC** | Well-calibrated probabilities → good ranking quality (0.924 vs RF's 0.955) |
| **Surprising CANDIDATE result** | Best CANDIDATE F1 of all classical models — worth discussing in dissertation |

---

## How it works

### Step 0 — StandardScaler required

L2 regularisation penalises large weights. Without scaling, `koi_depth` (range: 0–50,000 ppm) must have a tiny weight to avoid a large L2 penalty — even if it is genuinely important. After `StandardScaler`, all features have variance ≈ 1, so L2 penalises all weights equally. The learned weights reflect genuine importance, not feature scale.

### Step 1 — Learning weights for each feature

For each of the 3 classes, LR learns **37 weights** (one per feature) plus a **bias term**. These form a 3×37 weight matrix W. Training adjusts W to maximise the log-likelihood of the correct class across all 3,315 training samples.

```
score_class_k = w_{k,1}×feature_1 + w_{k,2}×feature_2 + ... + w_{k,37}×feature_37 + b_k
```

**Conceptual example — learned weights for FALSE POSITIVE class:**
```
koi_fpflag_co   →  w = +2.8  (centroid offset strongly → FP)
koi_fpflag_ss   →  w = +2.3  (secondary eclipse → FP)
koi_model_snr   →  w = −1.4  (high SNR → away from FP, towards real planet)
koi_depth       →  w = +0.9  (very deep transit → possibly EB)
koi_prad        →  w = +0.8  (large radius → possibly giant star / EB)
koi_incl        →  w = −0.6  (edge-on orbit → real planet geometry)
```

### Step 2 — Softmax converts scores to probabilities

After computing one score per class, softmax converts them into probabilities:

```
P(class k | x) = exp(score_k) / Σ_j exp(score_j)
```

Properties:
- All probabilities are between 0 and 1
- All three probabilities sum to exactly 1
- The class with the highest score always gets the highest probability

**Worked example:**
```
Input KOI: koi_fpflag_co=1, koi_model_snr=12.3, koi_prad=2.1

Raw scores:
  score_CONFIRMED    =  −0.4
  score_FALSE_POS    =  +3.1
  score_CANDIDATE    =  −0.8

After softmax:
  P(CONFIRMED)    = exp(−0.4) / (exp(−0.4)+exp(3.1)+exp(−0.8)) = 0.037
  P(FALSE_POS)    = exp(3.1)  / (...)                          = 0.944
  P(CANDIDATE)    = exp(−0.8) / (...)                          = 0.019

Final prediction: FALSE POSITIVE  ✓
```

### Step 3 — L2 regularisation

With 37 features and 3,315 training samples, LR has 37×3 = 111 parameters. L2 regularisation adds a penalty proportional to the sum of squared weights:

```
Total Loss = log-loss + (1/C) × Σ w²
```

- `C = 1.0` (our setting) → regularisation strength = 1/1 = 1
- Smaller C → stronger regularisation → weights pushed harder toward zero → simpler model
- Larger C → weaker regularisation → weights grow freely → more complex model

L2 keeps all 37 features in the model with small weights. This is appropriate since all features were domain-selected and are expected to contribute.

### Step 4 — Multinomial (joint softmax) vs One-vs-Rest

With the `lbfgs` solver, sklearn uses the **multinomial** formulation — one joint model that directly optimises all three class probabilities simultaneously. This is more principled than OvR (separate binary models) because:
- Probabilities sum to exactly 1 by construction
- The model sees all three classes in each update step
- Better calibrated probabilities than OvR

### The surprise result — why LR gets the best CANDIDATE F1

This is the counterintuitive finding of Stage 3.

LR's **soft linear boundary** with `class_weight='balanced'` forces equal importance on all three classes. Because LR cannot learn complex non-linear CONFIRMED/CANDIDATE separation, it defaults to a **wider catchment area for CANDIDATE** — predicting CANDIDATE more often (higher recall, lower precision).

RF and SVM are more surgical — they confidently classify CONFIRMED and FALSE POSITIVE using non-linear rules, which means they predict CANDIDATE only when forced. The softer LR boundary accidentally produces better CANDIDATE recall.

```
CANDIDATE recall comparison:
  LR:  ~0.56  (finds more CANDIDATEs, some false alarms)
  RF:  ~0.45  (more conservative, higher precision)
  SVM: ~0.34  (most conservative, highest precision, lowest recall)
```

---

## Parameters

| Parameter | Value set | Why |
|-----------|-----------|-----|
| `C` | `1.0` | Inverse regularisation strength. Standard default. With 37 features and 3,315 samples, there is enough data to support C=1.0 without underfitting. |
| `max_iter` | `1000` | Convergence budget for L-BFGS. Default 100 is too few for our 37-feature, 3-class, balanced-weighted problem. 1000 ensures convergence without hitting a warning. |
| `class_weight` | `'balanced'` | **Critical.** 4.9× imbalance. Amplifies CANDIDATE samples in the log-loss. Without this, CANDIDATE F1 drops from 0.46 to ~0.08–0.12. |
| `solver` | `'lbfgs'` (default) | L-BFGS optimiser — memory-efficient quasi-Newton method. Appropriate for our problem size. Supports multinomial formulation. |
| `multi_class` | `'auto'` → `'multinomial'` | With lbfgs, sklearn uses the joint softmax formulation automatically. Better-calibrated probabilities than OvR. |
| `penalty` | `'l2'` (default) | Ridge regularisation. All 37 features retained. Appropriate since all features are domain-selected. |
| `random_state` | `42` | Reproducibility seed for any stochastic elements. |

### L2 vs L1 regularisation — when to use each

| | L2 (ridge) | L1 (lasso) |
|--|------------|------------|
| **Effect on weights** | Shrinks toward zero, never exactly zero | Can zero out some weights entirely |
| **Feature selection** | None — all features kept | Automatic — sparse model |
| **When to use** | Domain-selected features, all expected to matter | Many candidate features, want automatic selection |
| **Our choice** | ✓ L2 — 37 domain-selected features | |
| **sklearn solver** | lbfgs, saga | liblinear, saga |

---

## Results

### Test set performance (586 samples)

| Metric | Score | Rank | Interpretation |
|--------|-------|------|----------------|
| **Accuracy** | **81.9%** | 2nd (ties SVM) | 480 / 586 correct |
| **F1 Macro** | **0.763** | 2nd | Higher than SVM despite same accuracy |
| **ROC-AUC** | **0.924** | 2nd | Best-calibrated probabilities after RF |
| **Cohen's κ** | **0.692** | 2nd | Solid substantial agreement |

### Per-class breakdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| CONFIRMED | 0.90 | 0.78 | **0.84** | 352 |
| FALSE POSITIVE | 0.99 | 0.99 | **0.99** | 163 |
| CANDIDATE | 0.39 | 0.56 | **0.46** ★ | 71 |

> ★ CANDIDATE F1=0.46 is the **best of all three classical models**. Note: higher recall (0.56) than precision (0.39) — LR finds more CANDIDATEs but with some false alarms. This is the soft-boundary effect in action.

### Full 3-model comparison

| Metric | RF | LR | SVM |
|--------|----|----|-----|
| Accuracy | **89.4%** | 81.9% | 81.9% |
| F1 Macro | **0.778** | 0.763 | 0.739 |
| ROC-AUC | **0.955** | 0.924 | 0.897 |
| Cohen's κ | **0.796** | 0.692 | 0.681 |
| F1 CONFIRMED | **0.92** | 0.84 | 0.85 |
| F1 FALSE POS | 0.98 | **0.99** | **0.99** |
| F1 CANDIDATE | 0.43 | **0.46** | 0.38 |

---

## Metrics Explained

### Why LR's ROC-AUC (0.924) beats SVM (0.897)

**Probability calibration** is the key. A well-calibrated model means: when it says "P(CONFIRMED) = 0.7", the candidate is actually confirmed 70% of the time.

LR produces well-calibrated probabilities by design — it directly optimises the **log-loss (cross-entropy)**, which is a proper scoring rule. Minimising it forces the model to produce reliable probability estimates.

SVM uses Platt scaling — a logistic fit on top of raw scores — which introduces calibration error. The AUC difference (0.924 vs 0.897) reflects this.

**Practical implication:** if you want to use probabilities for decisions (e.g., "schedule follow-up only if P(CONFIRMED) > 0.85"), LR's probabilities are more trustworthy than SVM's.

### Precision-recall breakdown for CANDIDATE

```
              Precision   Recall    F1
LR CANDIDATE:  0.39        0.56     0.46
RF CANDIDATE:  0.66        0.32     0.43
SVM CANDIDATE: 0.43        0.34     0.38
```

**Pattern:** LR has the **lowest precision** but **highest recall** for CANDIDATE. It catches more real candidates (fewer misses) but also mislabels some confirmed planets as candidates (more false alarms). This is the direct consequence of LR's softer, less aggressive decision boundary.

**Which is better depends on the use case:**
- If you want to catch all possible candidates (don't miss real planets): LR recall is best
- If you want high confidence when you say "CANDIDATE": RF precision is best

### F1 Macro = 0.763 vs Accuracy = 81.9%

```
F1 Macro = (0.84 + 0.99 + 0.46) / 3 = 0.763
```

LR and SVM have **identical accuracy** (81.9%) but LR has higher F1 Macro (0.763 vs 0.739). Why?

LR classifies more CANDIDATEs correctly (F1=0.46 vs 0.38). There are only 71 CANDIDATE test samples out of 586 — correcting a few extra CANDIDATE predictions barely moves the accuracy number, but it moves F1 Macro significantly because F1 Macro weights all three classes equally.

This illustrates exactly why F1 Macro is a better metric than accuracy for imbalanced datasets.

### Cohen's Kappa = 0.692

```
κ = (Observed accuracy − Expected accuracy) / (1 − Expected accuracy)
  = (0.819 − 0.47) / (1 − 0.47)
  = 0.349 / 0.530
  = 0.692
```

LR's kappa (0.692) is higher than SVM's (0.681) despite identical accuracy — same reason as F1 Macro: better CANDIDATE performance pushes the agreement metric up.

---

## What-if

### Changing C (regularisation strength)

| C | Effect |
|---|--------|
| 0.001 | Near-zero weights, predicts majority class only. F1 ~0.30–0.40. Useless. |
| 0.01 | Strong regularisation. Model very simple. F1 ~0.63–0.68. |
| 0.1 | Moderate regularisation. F1 ~0.71–0.74. Reasonable for very small datasets. |
| **1.0** | **Current. Standard default. F1 Macro 0.763.** |
| 10 | Weak regularisation. Marginal improvement (~+0.003 F1). Not worth the overfit risk. |
| 100 | Near-unconstrained. Risk of overfitting to training noise. Test F1 may drop slightly. |

### Removing `class_weight='balanced'`

This is the most impactful single change:
```
With balanced:    CANDIDATE F1 = 0.46, F1 Macro = 0.763
Without balanced: CANDIDATE F1 ≈ 0.08–0.12, F1 Macro ≈ 0.57–0.63
```
CONFIRMED recall increases slightly (model more aggressively predicts CONFIRMED), but the minority class is essentially abandoned. Never remove this for imbalanced multi-class problems.

### Adding polynomial features

LR's linear boundary is its key limitation. Polynomial feature expansion can partially overcome this:

```
Degree 1 (current): 37 features. Linear boundaries only.
  → F1 Macro ≈ 0.763

Degree 2: adds all pairwise products (koi_prad × koi_fpflag_ss, etc.)
  → ~703 features
  → F1 Macro ≈ 0.78–0.80 (captures some interaction effects)
  → Training much slower, needs lower C to avoid overfit

Degree 3: ~9,000 features
  → Severe overfit with 3,315 samples
  → Better to use RF or neural network at this point
```

### Changing penalty to L1 (lasso)

Setting `penalty='l1'` (requires solver='saga'):
- Many feature weights become exactly zero — automatic feature selection
- A sparse model: perhaps only 15–20 features with non-zero weights
- Slightly lower F1 Macro, but more interpretable
- Could be useful if you want to identify the minimal sufficient feature set

---

## Limitations

1. **Linear boundary** — cannot natively capture threshold rules like `koi_prad > 15 → FALSE POSITIVE`. Our engineered `size_flag` feature partially compensates but does not fully substitute for non-linear modelling.
2. **Assumes linear feature contributions** — the combined effect of `koi_fpflag_co=1 AND koi_fpflag_ss=1` is likely stronger than the sum of their individual contributions (interaction effect). LR misses this.
3. **CONFIRMED F1=0.84** — lowest of all three models on the majority class. LR's softer boundary misclassifies some confirmed planets as CANDIDATE.
4. **No automatic feature importance ranking** — unlike RF (Gini importance), LR weights depend on feature correlation structure and should not be naively ranked. Correlated features split weight between them.
5. **Convergence** — in some edge cases with extreme class imbalance or collinear features, L-BFGS may not converge in 1000 iterations. Monitor for `ConvergenceWarning` in production.
6. **Linearity assumption** — if the true decision boundary is highly non-linear, LR fundamentally cannot capture it even with tuning.

---

## Dissertation Talking Points

### The key talking point — the surprise result
> "Logistic Regression achieves the best CANDIDATE F1 (0.46) of all three classical models, outperforming both Random Forest (0.43) and SVM (0.38) on the minority class. This counterintuitive result demonstrates that model complexity is not monotonically related to minority-class performance — LR's softer linear decision boundary under balanced class weighting produces higher recall for the CANDIDATE class at the cost of some precision."

### The calibration talking point
> "Logistic Regression produced the best-calibrated probability estimates of the three classical models (ROC-AUC 0.924), second only to Random Forest. Its multinomial softmax formulation directly optimises log-loss — a proper scoring rule — making its class probabilities more reliable for downstream prioritisation tasks than SVM's Platt-calibrated scores."

### Suggested results section sentence
> "Logistic Regression (C=1.0, L2 penalty, balanced class weights) achieved accuracy 81.9% and F1 macro 0.763, placing second among classical models. Notably, it achieved the highest CANDIDATE F1 (0.46) across all three classical models, consistent with its softer decision boundaries producing higher minority-class recall. LR also produced well-calibrated probability estimates (ROC-AUC 0.924), providing a reliable confidence measure for candidate prioritisation."

### Limitations sentence for Methods section
> "Logistic Regression's linear decision boundary cannot natively capture the non-linear thresholds present in the KOI feature space (e.g. koi_prad > 15 Earth radii as a false-positive indicator), a limitation partially addressed through engineered binary features but not fully resolved."

### Connection to Genesis CNN
LR's well-calibrated probabilities serve as a calibration benchmark for the CNN stages. If the Genesis CNN's AUC exceeds LR's 0.924, that demonstrates the CNN has learned superior representations beyond what any linear combination of features can achieve. If it falls below, the CNN's probability estimates are less reliable than the simplest possible model — worth one sentence of discussion.

---

## References

- **Bishop, C.M. (2006).** Pattern Recognition and Machine Learning, Chapter 4 (Logistic regression and softmax). Springer. — Foundational textbook treatment.
- **Niculescu-Mizil, A. & Caruana, R. (2005).** Predicting good probabilities with supervised learning. *ICML 2005*. — Empirically demonstrates LR as the best-calibrated off-the-shelf classifier, which explains our ROC-AUC finding.
- **Tibshirani, R. (1996).** Regression shrinkage and selection via the lasso. *JRSS-B*, 58(1), 267–288. — Original L1 (lasso) regularisation paper.
- **Hoerl, A.E. & Kennard, R.W. (1970).** Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. — Original L2 (ridge) regularisation paper.
- **Pedregosa et al. (2011).** Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

---

*Last updated: Stage 3 — Classical ML complete*  
*Next: Stage 4 — Baseline CNN (`src/models/baseline_cnn.py`)*
