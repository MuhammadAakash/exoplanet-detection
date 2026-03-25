# 📚 Notes 06 — Logistic Regression

---

## The Name is Misleading

Despite having "Regression" in the name, Logistic Regression is a **classification** model. The name comes from the mathematical function it uses (the logistic/sigmoid function) — not because it predicts continuous numbers.

> Don't let the name confuse you. Logistic Regression answers: *"Which class does this belong to?"* — not *"What number is this?"*

---

## The Core Idea — A Weighted Vote

Logistic Regression assigns a **weight** to each feature and combines them into a single score.

### Real-World Analogy: Judging a Diving Competition

```
Feature                Weight    Meaning
─────────────────────────────────────────────────
Takeoff angle:    ×  0.3   →  moderate importance
Height reached:   ×  0.5   →  most important
Splash size:      × -0.4   →  negative (big splash = bad)
Entry angle:      ×  0.6   →  very important

Total score = (0.3 × takeoff) + (0.5 × height) + (-0.4 × splash) + (0.6 × entry)
```

### In Your Project

```python
score = (-3.2 × koi_fpflag_ss)     # secondary eclipse = bad sign
      + ( 0.8 × koi_max_mult_ev)   # strong repeating signal = good sign
      + (-2.1 × koi_fpflag_co)     # centroid offset = bad sign
      + ( 0.6 × koi_model_snr)     # high SNR = good sign
      + ( ... × other_features)
```

**Training** finds the weights that make these scores best match the actual labels across all 3,900 stars.

---

## The Sigmoid Function — Scores to Probabilities

The raw score can be any number. We need a probability between 0 and 1. The **sigmoid function** converts it:

```
           1
σ(score) = ────────────
           1 + e^(-score)
```

In plain terms:

```
Score    Probability    Interpretation
──────────────────────────────────────────
 -10  →    0.00005     Almost certainly negative class
  -5  →    0.007       Very unlikely positive class
  -2  →    0.12        Unlikely positive class
   0  →    0.50        50/50 — uncertain
  +2  →    0.88        Likely positive class
  +5  →    0.993       Very likely positive class
 +10  →    0.99995     Almost certainly positive class
```

The sigmoid squishes **any number** into a smooth S-shaped curve between 0 and 1.

```
Probability
   1 │                        ──────────────
     │                   ─────
     │              ─────
  0.5│─────────────x────────────────────────  ← 0.5 at score=0
     │        ─────
     │    ─────
   0 │──────────────────────────────────────
     ──────────────────────────────────────── Score
```

---

## For Multi-Class: Softmax

For 3 classes (CONFIRMED / FALSE POSITIVE / CANDIDATE), Logistic Regression uses **Softmax** — a generalisation of sigmoid for multiple classes.

```
For a new star, Logistic Regression computes 3 scores:
  Score_confirmed     = 2.3
  Score_false_positive = -1.1
  Score_candidate      = 0.4

After softmax:
  P(CONFIRMED)      = 0.72  → 72%
  P(FALSE POSITIVE) = 0.07  → 7%
  P(CANDIDATE)      = 0.21  → 21%

Sum = 1.0 ✓   Final prediction: CONFIRMED
```

---

## Training — How Weights Are Learned

### The Goal
Find weights that make the predicted probabilities as close as possible to the true labels.

### The Loss Function: Cross-Entropy
Measures how wrong the current predictions are.

```
If true label = CONFIRMED and model predicts:
  P(CONFIRMED) = 0.95  → Loss is LOW (good!)
  P(CONFIRMED) = 0.30  → Loss is HIGH (bad!)
```

Training minimises this loss by adjusting weights — using an algorithm called **gradient descent** (more on this in the CNN section).

---

## What the Weights Tell You

The learned weights are directly interpretable:

| Weight | Sign | Meaning |
|---|---|---|
| Large positive | ➕ | Feature strongly associated with this class |
| Large negative | ➖ | Feature strongly associated with OTHER classes |
| Near zero | ≈0 | Feature has little influence on prediction |

**Example interpretation for CONFIRMED class:**
```
koi_fpflag_ss:   weight = -4.2  → If flag=1, very unlikely CONFIRMED
koi_max_mult_ev: weight = +1.8  → Higher MES → more likely CONFIRMED
koi_prad:        weight = +0.3  → Larger planet → slightly more likely CONFIRMED
koi_fpflag_nt:   weight = -2.9  → If flag=1, very unlikely CONFIRMED
```

This is scientifically meaningful — it matches exactly what astronomers know about planet vetting.

---

## The C Parameter — Regularisation

Just like SVM, Logistic Regression has a C parameter:

```
Low C (e.g. 0.01):
  - Heavily penalises large weights
  - Forces simpler model
  - Better generalisation, lower accuracy

High C (e.g. 100):
  - Allows large weights
  - More flexible model
  - Risk of overfitting

In your project: C=1.0 (default — balanced)
```

**Why regularisation?** Without it, Logistic Regression might give huge weights to features that happen to correlate with the label in training data by chance — and fail on new data.

---

## Logistic Regression Needs StandardScaler Too

Just like SVM, Logistic Regression is sensitive to feature scale.

**Without scaling:**
- `koi_depth` (range: 50–500,000 ppm) dominates
- `koi_impact` (range: 0–1) becomes irrelevant
- The model can't find appropriate weights for both simultaneously

**With StandardScaler:**
- All features on same scale
- Weights are comparable and meaningful
- Better, more stable training

---

## Results in Your Project

| Metric | Logistic Regression | Why |
|---|---|---|
| Accuracy | 80.3% | Lowest classical model — linear boundaries not flexible enough |
| F1 Macro | 0.718 | Struggles with CANDIDATE class minority |
| ROC-AUC | 0.911 | Still good at ranking — linear model captures main patterns |
| Cohen's κ | 0.702 | Decent agreement, but room for improvement |

**Why is it lowest?** Logistic Regression assumes a **linear decision boundary** — a straight flat surface separating the classes. Real astrophysical data has complex, non-linear relationships that a straight surface can't capture.

> However: 80.3% with a linear model is actually impressive. It means the features you selected carry strong, linearly separable signal. If LR scored 50%, you'd worry about your features.

---

## Logistic Regression vs The Others

| Aspect | Logistic Regression | SVM | Random Forest |
|---|---|---|---|
| Decision boundary | Linear (flat) | Non-linear (curved) | Complex (many splits) |
| Interpretability | ✅ Best — weights are direct | ❌ Hard | ⚠️ Feature importance only |
| Needs scaling? | ✅ Yes | ✅ Yes | ❌ No |
| Training speed | ✅ Fastest | ❌ Slowest | ⚠️ Medium |
| Performance | ⚠️ Good | ✅ Better | ✅ Best (classical) |
| Best used for | Baseline + interpretation | Well-separated data | Most tabular problems |

---

## Why Include It If It's the Weakest?

1. **Baseline reference** — if Genesis CNN only beats LR by 1%, the CNN complexity isn't justified
2. **Interpretability** — weights tell you *why* the model made each decision
3. **Speed** — train in seconds for a quick sanity check
4. **Probability calibration** — its probability outputs are the most trustworthy of the three

> **A well-functioning ML pipeline should show: LR < SVM < RF < CNN.** If this ordering breaks, something is wrong (data leakage, bugs, etc.)

---

## Key Concepts Summary

```
Weights (Coefficients)
    └── How much each feature influences the prediction
        — learned during training

Sigmoid Function
    └── Converts any score to a probability between 0 and 1

Softmax
    └── Multi-class version of sigmoid
        — outputs probabilities for all classes that sum to 1

Cross-Entropy Loss
    └── Measures how wrong the current predictions are
        — training minimises this

Regularisation (C parameter)
    └── Prevents overfitting by penalising large weights
        — C=1.0 in your project

Linear Decision Boundary
    └── LR separates classes with a flat surface
        — works well only if classes are linearly separable
```

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **Logistic Regression** | Linear classifier using weighted sum + sigmoid |
| **Weight / Coefficient** | How much influence a feature has on the prediction |
| **Sigmoid** | S-shaped function mapping any number to 0–1 range |
| **Softmax** | Multi-class version of sigmoid — probabilities sum to 1 |
| **Cross-Entropy** | Loss function measuring how wrong probability predictions are |
| **Regularisation** | Penalising large weights to prevent overfitting |
| **C parameter** | Controls regularisation strength (low C = more regularisation) |
| **Linear boundary** | Flat surface separating classes — LR's key limitation |
| **Gradient Descent** | Algorithm for finding the best weights during training |
| **Interpretability** | Ability to understand why a model made a decision |

---

*Notes from MSc Data Science Dissertation — Exoplanet Candidate Vetting*
