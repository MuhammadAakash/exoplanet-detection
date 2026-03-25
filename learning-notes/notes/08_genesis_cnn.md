# 📚 Notes 08 — Genesis CNN

---

## What is Genesis CNN?

Genesis CNN is a **simplified, sparse Convolutional Neural Network** for exoplanet detection, proposed by:

> Visser, K., Bosma, B., & Postma, E. (2022).
> *"Exoplanet detection with Genesis."*
> Journal of Astronomical Instrumentation, 11(3).
> arXiv: 2105.06292

It was designed to answer the question: *"Do we actually need a massive, complex network to detect exoplanets — or can a much simpler one do just as well?"*

**Answer: A simpler one does just as well — and often better.**

---

## The Context: Where Genesis Sits in the Timeline

```
AstroNet (Shallue & Vanderburg, 2018)
    └── First CNN for exoplanet vetting
    └── 2 branches: global view + local view of light curve
    └── ~10,000,000 parameters
    └── 96% accuracy on Kepler DR24

Exonet-XS (Ansdell et al., 2018)
    └── Added stellar parameters + centroids
    └── ~850,000 parameters
    └── 97.5% accuracy (better with domain knowledge)

Genesis (Visser et al., 2022) ← YOUR MODEL
    └── Single branch, one view only
    └── ~390,000 parameters (>95% fewer than AstroNet)
    └── Outperforms both AstroNet and Exonet-XS in fair comparison
    └── Key finding: simpler = better for this task
```

---

## Pipeline 1: Tabular Genesis CNN (Your Adaptation)

This is your primary model — applied to the **37 tabular KOI features**, not raw light curves.

### Architecture: Dual-Branch

```
Input (37 features, 1 channel)
        │
        ├─────────────────────────────┐
        │  LOCAL BRANCH               │  GLOBAL BRANCH
        │  Conv1D(32, k=3) + BN + ReLU│  Conv1D(32, k=7) + BN + ReLU
        │  Conv1D(64, k=3) + BN + ReLU│  Conv1D(64, k=5) + BN + ReLU
        │  GlobalAveragePooling → (64,)│  GlobalAveragePooling → (64,)
        └──────────────┬──────────────┘
                       │
                  Concatenate → (128,)
                       │
              Dense(128, ReLU) + Dropout(0.4)
                       │
              Dense(64,  ReLU) + Dropout(0.4)
                       │
              Dense(3, Softmax) → CONFIRMED / FALSE POS / CANDIDATE
```

### Why Two Branches?

**Local Branch (kernel=3):** Detects interactions between features that are *adjacent* in the feature vector.

```
Features are ordered: [transit features] [stellar features] [flags] [signal] [magnitudes]

Within transit group:
  koi_period, koi_time0bk, koi_impact, koi_duration, koi_depth, ...
  ↑────────────────────────k=3 window──────────────────────────↑

A kernel of size 3 slides across and detects:
  "depth + duration + radius ratio together"
  "impact + inclination + period together"
```

**Global Branch (kernel=7):** Detects interactions across a wider window — spanning across feature groups.

```
  [transit features]     [stellar features]
  koi_depth, koi_ror, koi_srho, koi_steff, koi_slogg, koi_smet, koi_srad
  ↑────────────────────────k=7 window───────────────────────────────────↑

"Transit depth + stellar temperature + surface gravity together"
→ This combination catches giant star contamination cases
```

### Why Not Just Use One Branch?

The Baseline CNN (Stage 4) only used kernel=3. It got 85.5% — 3.9% below Random Forest.

The hypothesis: **some patterns require looking at multiple features together across a wider window.** The global branch (k=7) captures these wider patterns that the local branch (k=3) misses.

Result: Genesis CNN (~91%) beats both Baseline CNN and Random Forest.

---

## Pipeline 2: Light Curve Genesis CNN (Visser 2022)

This is the **faithful implementation** of the original Genesis paper — applied to raw Kepler flux data.

### Architecture: Single-Branch

```
Input (3,197 flux points, 1 channel)
        │
Conv1D(16 filters, kernel=16) + ReLU → MaxPool(size=4, stride=2)
        │
Conv1D(32 filters, kernel=8)  + ReLU → MaxPool(size=4, stride=2)
        │
Conv1D(64 filters, kernel=4)  + ReLU → MaxPool(size=2, stride=2)
        │
Conv1D(128 filters, kernel=2) + ReLU → MaxPool(size=2, stride=2)
        │
Flatten → Dense(512, ReLU) → Dropout(0.1)
        │
Dense(1, Sigmoid) → Planet / Non-Planet
```

### Key Design Decisions Explained

**Why start with a large kernel (k=16)?**

A transit event spans many time steps. If the orbital period is 9 days and Kepler takes one measurement every 30 minutes, a 3-hour transit spans 6 flux measurements. A kernel of size 16 can see the whole transit shape at once.

```
Flux: [0.01, 0.02, -0.12, -0.85, -1.00, -0.85, -0.12, 0.02, 0.01, ...]
      ←─────────────── kernel=16 can see the full dip ───────────────→
      ← k=3 only sees three adjacent points — misses the dip shape →
```

**Why does the kernel shrink (16→8→4→2)?**

Each MaxPool layer compresses the sequence. By the time we're 3 layers deep, the sequence is much shorter — each position represents a larger chunk of the original. A small kernel at this point covers a wide original range.

**Why only Dropout(0.1) — so little?**

The dataset only has 37 planet examples (in training). Very sparse model = less overfitting risk. Visser et al. found that 0.1 was optimal — more dropout reduced performance.

**Why sigmoid (not softmax)?**

Binary task (planet vs non-planet) → one output neuron → sigmoid gives P(planet).

---

## Training Tricks for Severe Imbalance

The light curve dataset has a 136:1 imbalance. This requires special handling.

### Class Weights

```python
class_weight = {
    0: 0.507,   # Non-planet gets weight ~0.5 (less important)
    1: 68.7,    # Planet gets weight ~69  (136× more important!)
}
```

Each planet example now contributes as much to the loss as 136 non-planet examples.

### Monitor AUC, Not Accuracy

```
A model that always predicts "No Planet":
  Accuracy = 99.3%  ← looks amazing!
  ROC-AUC  = 0.50   ← random guessing ← the real story

A good model:
  Accuracy = 95%
  ROC-AUC  = 0.97   ← genuinely distinguishing planets
```

Always monitor ROC-AUC (or F1-Planet) when dealing with severe imbalance.

### EarlyStopping on val_auc

```python
EarlyStopping(monitor="val_auc", patience=10, mode="max")
```

Stop when validation AUC stops improving — not when loss stops.

### ReduceLROnPlateau

If AUC plateaus for 5 epochs, halve the learning rate. This gives the model a second chance to fine-tune with smaller steps.

```
LR schedule:
  Epochs 1-15: lr = 1e-4
  Epochs 16-25: lr = 5e-5  (halved — AUC plateaued)
  Epochs 26-30: lr = 2.5e-5 (halved again)
  Epoch 35: EarlyStopping fires
```

---

## Genesis vs AstroNet vs Baseline CNN

| | Baseline CNN (your Stage 4) | Tabular Genesis CNN (your Stage 5) | LC Genesis CNN (Visser 2022) |
|---|---|---|---|
| Input | 37 tabular features | 37 tabular features | 3,197 flux points |
| Branches | 1 | 2 (local + global) | 1 |
| Kernel sizes | 3, 3, 3 | Local: 3,3 / Global: 7,5 | 16, 8, 4, 2 |
| Parameters | ~50K | ~80K | ~390K |
| AstroNet params | — | — | ~10,000,000 |
| Task | 3-class | 3-class | Binary |
| Accuracy | 85.5% | ~91% | ~95%+ |
| Key innovation | Basic deep learning baseline | Multi-scale feature capture | Sparse architecture |

---

## Why Genesis Outperforms More Complex Models

> **Key finding from Visser et al. 2022:** *"Existing exoplanet detection CNNs are too complex for the task at hand."*

For a dataset of ~5,000 stars (small by deep learning standards):
- 10,000,000 parameters → massive overfitting risk
- 390,000 parameters → appropriate for dataset size
- Sparse model → forced to learn general, robust features

> **Analogy:** Teaching a 5-year-old child using a university textbook vs a picture book. The picture book is appropriate for the level of data available. The university textbook is overkill and confusing.

---

## The Evaluation Metrics Explained

### For Tabular Pipeline (3-class)

| Metric | Formula | When to Use |
|---|---|---|
| **Accuracy** | Correct / Total | Rough overall measure |
| **F1 Macro** | Average F1 across all 3 classes | Imbalanced data — penalises ignoring minority class |
| **ROC-AUC** | Area under ROC curve | Ranking quality |
| **Cohen's κ** | Accuracy minus chance | More honest than raw accuracy |
| **MCC** | Matthews Correlation Coef | Best single metric for imbalanced multi-class |

### For Light Curve Pipeline (binary)

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **ROC-AUC** | Can model rank planets above non-planets? | Primary metric — unaffected by threshold |
| **F1-Planet** | Of all actual planets, how many found? | Critical — missing a real planet is costly |
| **Precision** | Of predicted planets, how many are real? | Controls false alarms |
| **Recall** | Of all real planets, how many found? | Most important for discovery context |
| **MCC** | Overall quality — handles 136:1 imbalance | Best summary metric |

---

## Key Concepts Summary

```
Genesis CNN
    └── Sparse, simplified CNN — fewer parameters, better generalisation

Two Implementations in This Project:
    ├── Tabular Genesis: dual-branch on 37 features (3-class)
    └── LC Genesis:      single-branch on 3,197 flux points (binary)

Local Branch (k=3)
    └── Detects interactions between adjacent features

Global Branch (k=7)
    └── Detects patterns spanning wider feature windows

Weight Sharing (Conv)
    └── Same kernel applied everywhere — drastically fewer parameters

MaxPooling
    └── Keeps strongest activations, compresses sequence

EarlyStopping
    └── Stops training when validation metric stops improving

Class Weights
    └── Makes rare class (planet/candidate) count more during training

ROC-AUC
    └── Best metric when data is severely imbalanced
```

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **Genesis CNN** | Sparse single-branch CNN — Visser et al. (2022) |
| **Dual-branch** | Two parallel conv branches capturing different pattern scales |
| **Local branch** | Small kernel — detects adjacent feature interactions |
| **Global branch** | Large kernel — detects wide, cross-group patterns |
| **Weight sharing** | Same kernel weights reused at every position |
| **Sparse architecture** | Fewer parameters — reduces overfitting, improves generalisation |
| **ROC-AUC** | Area under ROC curve — measures ranking quality (ideal for imbalance) |
| **F1-Planet** | F1 score specifically for the planet class |
| **Recall** | Of all real positives, how many did the model find? |
| **Precision** | Of all predicted positives, how many are actually positive? |
| **MCC** | Matthews Correlation Coefficient — robust metric for imbalanced data |
| **EarlyStopping** | Halt training when validation metric stops improving |
| **ReduceLROnPlateau** | Halve learning rate when validation plateaus |
| **class_weight='balanced'** | Automatically weight rare classes proportionally |

---

*Notes from MSc Data Science Dissertation — Exoplanet Candidate Vetting*
