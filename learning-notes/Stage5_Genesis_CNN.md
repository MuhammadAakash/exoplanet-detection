# Genesis CNN — Complete Reference Notes
> Exoplanet Candidate Vetting · Stage 5 · MSc Data Science Dissertation
> Dataset: NASA Kepler KOI Q1-Q17 DR25 · 3,901 samples · 37 features · 3 classes

---

## Table of Contents
1. [What is the Genesis CNN?](#what-is-the-genesis-cnn)
2. [Why dual branches?](#why-dual-branches)
3. [How the Genesis CNN works — step by step](#how-it-works)
4. [Every component explained](#every-component)
5. [Parameters and why we chose them](#parameters)
6. [Training process explained](#training-process)
7. [Results on our dataset](#results)
8. [vs Baseline CNN and Classical ML](#vs-prior-models)
9. [Limitations](#limitations)
10. [Dissertation talking points](#dissertation-talking-points)

---

## What is the Genesis CNN?

The Genesis CNN is a **dual-branch** convolutional neural network. Instead of one sequence of Conv1D layers (as in the Baseline CNN), it runs two parallel branches simultaneously — one with small kernels and one with large kernels — then merges both into a single classification head.

The name "Genesis" is inspired by the architecture used in Shallue & Vanderburg (2018), who used two parallel CNN branches to process both the global and local views of a Kepler light curve. Our version applies the same dual-branch concept to **pre-extracted tabular features** rather than raw time-series data.

---

## Why Dual Branches?

### The problem with the Baseline CNN

The Baseline CNN used `kernel_size=3` in all Conv1D layers. This means every filter looks at a window of exactly 3 adjacent features at a time. It can learn:
- What pairs or triplets of adjacent features tend to co-occur
- e.g., high SNR + high depth + short duration → transit-like

But it cannot learn:
- How transit geometry features (positions 1–13) relate to signal quality features (positions 30–35)
- How a combination of stellar temperature + orbital period + insolation flux interact

These **long-range interactions** require a larger receptive field.

### The hypothesis

> Adding a second parallel branch with larger kernels (`kernel_size=7, 5`) will capture multi-scale feature interactions that the single-branch Baseline CNN misses — potentially closing the ~3.9% accuracy gap with Random Forest.

### Why not just use larger kernels everywhere?

One option is to replace `kernel_size=3` with `kernel_size=7` in all blocks. But this would:
1. Force the model to use only the wider receptive field, losing the local detail
2. Increase parameter count without capturing both scales simultaneously

The dual-branch design keeps **both** receptive fields active, letting the classifier see the same input through two different lenses. This is the core architectural insight.

---

## How the Genesis CNN Works

### Full pipeline

```
Raw input:  37 pre-extracted KOI features, already StandardScaled (from Stage 1)

Step 1: Reshape
  (N, 37) → (N, 37, 1)  — same as Baseline CNN

Step 2a: LOCAL BRANCH (kernel_size = 3, 3)
  Block 1: Conv1D(32, k=3) → BatchNorm → ReLU    shape: (37, 32)
  Block 2: Conv1D(64, k=3) → BatchNorm → ReLU    shape: (37, 64)
  GlobalAveragePooling1D                          shape: (64,)

Step 2b: GLOBAL BRANCH (kernel_size = 7, 5)
  Block 1: Conv1D(32, k=7) → BatchNorm → ReLU    shape: (37, 32)
  Block 2: Conv1D(64, k=5) → BatchNorm → ReLU    shape: (37, 64)
  GlobalAveragePooling1D                          shape: (64,)

Step 3: Concatenate both branches
  (64,) + (64,) → (128,)

Step 4: Dense head
  Dense(128, ReLU) → Dropout(0.4)
  Dense(64,  ReLU) → Dropout(0.4)
  Dense(3, Softmax)

Final output:
  [P(CONFIRMED), P(FALSE POSITIVE), P(CANDIDATE)]
```

### Visualised

```
Input: 37 features (one KOI candidate)
──────────────────────────────────────────────────────────────
         LOCAL BRANCH                  GLOBAL BRANCH
  ┌──────────────────────┐    ┌──────────────────────┐
  Conv1D(32, k=3)+BN+ReLU    Conv1D(32, k=7)+BN+ReLU
  Conv1D(64, k=3)+BN+ReLU    Conv1D(64, k=5)+BN+ReLU
  GlobalAvgPool → (64,)       GlobalAvgPool → (64,)
  └──────────────────────┘    └──────────────────────┘
──────────────────────────────────────────────────────────────
                  Concatenate → (128,)
                       │
          Dense(128, ReLU) + Dropout(0.4) → (128,)
                       │
          Dense(64,  ReLU) + Dropout(0.4) → (64,)
                       │
          Dense(3, Softmax) → (3,)
──────────────────────────────────────────────────────────────
  [0.81, 0.04, 0.15]  →  predicted: CONFIRMED
```

---

## Every Component Explained

### Why `kernel_size=7` for the global branch?

The first global block uses `kernel_size=7`. This means the filter looks at 7 adjacent features simultaneously. With 37 features total, a window of 7 covers ~19% of the feature space at each position — enough to capture interactions between features that belong to different physical groups.

With `padding="same"`, the output remains shape `(37, 32)`, so no information is lost at the edges.

### Why `kernel_size=5` for the second global block?

The second global block uses `kernel_size=5` (down from 7). This is a common pattern: start with a larger receptive field, then narrow it slightly to build more abstract representations from the wider features learned in block 1. It avoids having the model over-smooth across too many features in the final convolutional stage.

### Why GAP after each branch (not after concatenation)?

GlobalAveragePooling1D collapses `(37, 64)` → `(64,)` within each branch before merging. The alternative would be to concatenate the raw Conv1D outputs before pooling, giving shape `(37, 128)` → then GAP → `(128,)`. Both approaches work, but pooling per-branch is cleaner because:
- Each branch produces an independent 64-d summary of what it found
- The classifier can then combine these summaries, rather than pooling a mixed tensor where local and global patterns are already interleaved

### Why Concatenate, not Add?

Adding the two branch vectors (`loc + glob`) would require them to represent the same features in the same vector space. That's a strong constraint — it forces local and global patterns to align dimension-by-dimension.

Concatenation (`[loc, glob]`) makes no such assumption. The Dense head receives all 128 values and learns which combinations matter. It is strictly more expressive than addition.

### Why `Dense(128)` then `Dense(64)` — a two-layer head?

The Baseline CNN used a single `Dense(64)` head. The Genesis CNN's merged vector is 128-d (double the Baseline's GAP output), so the head is sized to match:
- `Dense(128)` first reduces the 128-d merge without discarding information too fast
- `Dense(64)` then compresses to the same scale as the Baseline's head before the output layer

This gives the classifier sufficient capacity to learn non-linear combinations of local + global features without collapsing them in a single aggressive step.

### Why `Dropout(0.4)` instead of `0.3`?

The Genesis CNN has more parameters and more capacity than the Baseline CNN. A slightly higher dropout rate (0.4 vs 0.3) provides stronger regularization to prevent the larger model from overfitting on the same 2,730-sample training set.

---

## Parameters

### Architecture config (`CNN_CONFIG` in `config.py`)

```python
CNN_CONFIG = {
    "b1_filters"   : [32, 64],    # Local branch filters (per block)
    "b1_kernels"   : [3, 3],      # Local branch kernel sizes
    "b2_filters"   : [32, 64],    # Global branch filters (per block)
    "b2_kernels"   : [7, 5],      # Global branch kernel sizes
    "dense_units"  : [128, 64],   # Dense head units
    "dropout_rate" : 0.4,         # Dropout after each Dense layer
    "num_classes"  : 3,           # CONFIRMED / FALSE POSITIVE / CANDIDATE
    "epochs"       : 60,
    "batch_size"   : 32,
    "learning_rate": 1e-3,
    "patience"     : 10,
    "l2_lambda"    : 1e-4,
}
```

### Parameter count breakdown

```
LOCAL BRANCH
  Conv1D(32, k=3):    3 × 1 × 32  =    96 weights
  Conv1D(64, k=3):    3 × 32 × 64 = 6,144 weights
  BatchNorm ×2:       ≈ 384 params

GLOBAL BRANCH
  Conv1D(32, k=7):    7 × 1 × 32  =   224 weights
  Conv1D(64, k=5):    5 × 32 × 64 = 10,240 weights
  BatchNorm ×2:       ≈ 384 params

DENSE HEAD
  Dense(128):         128 × 128 + 128  = 16,512 params
  Dense(64):          128 × 64  + 64   =  8,256 params
  Dense(3):           64  × 3   + 3    =    195 params
─────────────────────────────────────────────────────────
Total: ~42,435 parameters   (still a tiny model, ~166 KB)
```

The Genesis CNN is only ~2,300 parameters larger than the Baseline CNN. The architectural change (dual branches) adds very little overhead while potentially providing significantly richer representations.

---

## Training Process

The training setup is identical to the Baseline CNN for a fair comparison:

- **Optimizer:** Adam (lr=1e-3)
- **Loss:** Sparse categorical cross-entropy
- **Class weights:** Balanced (CONFIRMED: 0.556, FALSE POSITIVE: 1.201, CANDIDATE: 2.725)
- **EarlyStopping:** patience=10, monitor=val_loss, restore_best_weights=True
- **ReduceLROnPlateau:** factor=0.5, patience=5, min_lr=1e-6
- **ModelCheckpoint:** saves best val_loss weights to `genesis_cnn_best.keras`

The only differences in training are:
1. `Dropout(0.4)` instead of `0.3` (more regularization for larger capacity)
2. Two-layer Dense head instead of one

---

## Results

*(To be filled in after training)*

| Metric | Baseline CNN | Genesis CNN | Change |
|--------|-------------|-------------|--------|
| Accuracy | 85.5% | ? | ? |
| F1 Macro | 0.754 | ? | ? |
| ROC-AUC | 0.911 | ? | ? |
| Cohen's κ | 0.732 | ? | ? |
| F1 CONFIRMED | 0.89 | ? | ? |
| F1 FALSE POS | 0.98 | ? | ? |
| F1 CANDIDATE | 0.39 | ? | ? |
| Parameters | 40,163 | ~42,435 | +2,272 |

---

## vs Prior Models

### Does the dual branch close the gap with Random Forest?

This is the central question of Stage 5. The Baseline CNN trailed RF by 3.9% accuracy. The Genesis CNN's dual-branch design directly tests whether multi-scale feature processing recovers this gap.

Possible outcomes and their interpretation:

| Outcome | Interpretation |
|---------|----------------|
| Genesis CNN > RF | Multi-scale feature interactions are the missing piece; the CNN architecture assumption is validated |
| Genesis CNN ≈ Baseline CNN | The gap is not architectural — it reflects a fundamental tabular data advantage of tree-based models |
| Genesis CNN < Baseline CNN | The larger model overfits on 2,730 samples; the Baseline's simplicity was an advantage |

### Why RF still has inherent advantages on tabular data

Even with dual branches, the Genesis CNN faces the same structural challenges as the Baseline CNN on tabular data:
1. Feature ordering is arbitrary — adjacency in the vector is not physically meaningful
2. The false-positive flags (`koi_fpflag_*`) have binary, threshold-based discriminative power that a tree split handles in one step; the CNN must learn this through gradient descent
3. 2,730 training samples is a regime where tree-based methods have historically dominated (Grinsztajn et al., 2022)

---

## Limitations

1. **Feature ordering is still arbitrary.** Both branches exploit adjacency in the feature vector, but this adjacency was set by the column order in the original CSV. The "local" and "global" patterns the CNN learns are real but depend on this ordering choice.

2. **No attention mechanism.** Unlike a Transformer, the Genesis CNN cannot learn to attend to specific feature pairs regardless of position. An attention layer would be a natural extension.

3. **Two branches only.** A three-branch design (e.g., k=3, k=5, k=7) would capture even more scales. Not implemented here for simplicity and to limit overfitting risk.

4. **CANDIDATE class remains hard.** No architectural change can fix the fact that CANDIDATE is an administratively-labelled class (unresolved cases), not a physically distinct class. Expect F1 ≈ 0.3–0.45 regardless of architecture.

5. **Still a tabular model.** The full Shallue & Vanderburg (2018) Genesis architecture operates on raw light curves — ~2,000 time-steps of actual photometric signal. Our adaptation to 37 tabular features is a simplification that loses the temporal structure of the transit signal.

---

## Dissertation Talking Points

### Lead result
> "The Genesis CNN achieves [result] on the held-out test set, [compared to] the Baseline CNN's 85.5% and Random Forest's 89.4%. The dual-branch design [does/does not] close the performance gap attributable to single-scale feature processing."

### Why this architecture is interesting
> "The Genesis CNN tests a specific hypothesis: that multi-scale feature interactions — captured by running parallel Conv1D branches with kernel_size=3 and kernel_size=7 — are the key limitation of the single-branch Baseline CNN. The architecture is directly inspired by Shallue & Vanderburg (2018), adapted from raw light curves to pre-extracted tabular features."

### The question your examiners will ask
**"Why use dual branches? Why not just use a larger single kernel?"**

Two reasons:
1. *Expressiveness:* A single large kernel (k=7) would capture wide interactions but lose local detail. The dual branch captures both scales simultaneously. The concatenated representation is strictly more informative.
2. *Experimental design:* Using dual branches isolates the architectural variable. Comparing Baseline CNN vs Genesis CNN tells us specifically whether multi-scale processing matters, not just whether a bigger or different model helps.

---

## References

- **Shallue, C.J. & Vanderburg, A. (2018).** Identifying Exoplanets with Deep Learning. *AJ*, 155(2). — Original Genesis CNN architecture using dual-branch Conv1D on raw Kepler light curves.
- **Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022).** Why tree-based models still outperform deep learning on tabular data. *NeurIPS 2022*.
- **He, K. et al. (2016).** Deep Residual Learning for Image Recognition. *CVPR 2016*. — Dual-path / multi-scale CNN design principles.

---

*Last updated: Stage 5 — Genesis CNN implemented*
*Next: Stage 6 — Final Comparison (`src/evaluation/compare_models.py`)*
