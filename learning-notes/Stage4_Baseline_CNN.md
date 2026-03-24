# Baseline CNN — Complete Reference Notes
> Exoplanet Candidate Vetting · Stage 4 · MSc Data Science Dissertation  
> Dataset: NASA Kepler KOI Q1-Q17 DR25 · 3,901 samples · 37 features · 3 classes

---

## Table of Contents
1. [What is a Convolutional Neural Network?](#what-is-a-cnn)
2. [Why apply a CNN to tabular data?](#why-cnn-on-tabular)
3. [How the Baseline CNN works — step by step](#how-it-works)
4. [Every component explained](#every-component)
5. [Parameters and why we chose them](#parameters)
6. [Training process explained](#training-process)
7. [Results on our dataset](#results)
8. [vs Classical ML — the honest comparison](#vs-classical-ml)
9. [What-if: changing things](#what-if)
10. [Limitations](#limitations)
11. [Dissertation talking points](#dissertation-talking-points)
12. [References](#references)

---

## What is a Convolutional Neural Network?

A CNN is a type of neural network that uses **filters** (also called kernels) which slide across the input and look for patterns at each position. Originally invented for images — where a filter might detect edges or textures — the same idea works for any sequential data.

Instead of connecting every input to every neuron (which is what a Dense/fully-connected layer does), a Conv1D layer applies a small window of weights across the input. This means:
- It learns **local patterns** — what combinations of nearby features tend to occur together
- It shares weights across positions — the same filter is reused, which means fewer parameters
- It is **translation-invariant** in a limited sense — a pattern detected at position 5 uses the same weights as the same pattern at position 12

> **One-line intuition:** Imagine dragging a magnifying glass across your 37 features, three at a time, asking "is there an interesting pattern in this trio?" That's a Conv1D with kernel_size=3.

---

## Why Apply a CNN to Tabular Data?

This is a genuinely interesting methodological question, and it's worth being able to answer it confidently in a viva.

Traditional CNNs process images, audio, or text — data with natural spatial or temporal structure. Our KOI features are tabular (a row of 37 numbers). There is no inherent "position 1 is next to position 2" relationship in the same way a pixel is next to another pixel. So why bother with Conv1D?

**The argument for using it:**

Most of our 37 features are not independent — they are correlated in astrophysically meaningful ways. For example:

- `koi_fpflag_co` and `koi_fpflag_ss` both signal false positives and tend to co-occur
- `koi_model_snr`, `koi_depth`, and `koi_duration` are all related to transit geometry
- Stellar parameters (`koi_steff`, `koi_srad`, `koi_smass`) form a coherent physical group

When we reshape our 37 features from shape `(37,)` to `(37, 1)` and pass them through a Conv1D with kernel_size=3, the model can learn **what combinations of adjacent features are informative**. This is different from a Dense layer, which treats all 37 features as equally independent.

**The honest caveat:**

The feature ordering is somewhat arbitrary — it was set during preprocessing, not by any physical law. So the "adjacency" the CNN exploits is partially constructed. This is why we:
1. Frame this as a **baseline** — a test of whether CNNs are competitive at all on this data
2. Use it as a stepping stone to the dual-branch Genesis CNN (Stage 5), which is the real architectural contribution

**The practical justification:**

We are following the spirit of Shallue & Vanderburg (2018), who used Conv1D on raw light curve data. Our simplified adaptation applies the same architecture family to pre-extracted features. The question our dissertation investigates is: does the architectural insight (dual receptive fields, in Stage 5) survive even when we operate on tabular features rather than raw signal?

---

## How the Baseline CNN Works

### The full pipeline

```
Raw input:  37 pre-extracted KOI features, already StandardScaled (from Stage 1)

Step 1: Reshape
  (2730, 37)  →  (2730, 37, 1)
  We add a channel dimension so Conv1D can operate on it.
  Think of it as: 37 time-steps, each with 1 channel (like mono audio vs stereo).

Step 2: Three convolutional blocks
  Block 1: Conv1D(32 filters, kernel=3) → BatchNorm → ReLU
  Block 2: Conv1D(64 filters, kernel=3) → BatchNorm → ReLU
  Block 3: Conv1D(128 filters, kernel=3) → BatchNorm → ReLU

  Each block DOUBLES the filters, building richer representations.
  Shape stays (37, filters) throughout — padding="same" keeps length constant.

Step 3: GlobalAveragePooling1D
  (37, 128)  →  (128,)
  Takes the average across all 37 positions for each of the 128 filters.
  This collapses the spatial dimension into a single vector of 128 numbers.

Step 4: Dense head
  Dense(64, ReLU) → Dropout(0.3) → Dense(3, softmax)
  Standard classification head. Outputs three probabilities that sum to 1.

Final output:
  [P(CONFIRMED), P(FALSE POSITIVE), P(CANDIDATE)]
  Predicted class = argmax of these three probabilities.
```

### Visualised

```
Input: 37 features (one KOI candidate)
─────────────────────────────────────────────────────
 f1   f2   f3  ...  f35  f36  f37
  │    │    │         │    │    │
  └────┴────┘         └────┴────┘
    kernel=3            kernel=3
  (sliding window,    (sliding window,
   32 filters)         32 filters)
─────────────────────────────────────────────────────
After Block 1:  shape (37, 32)   — 32 learned patterns
After Block 2:  shape (37, 64)   — 64 richer patterns
After Block 3:  shape (37, 128)  — 128 complex patterns
─────────────────────────────────────────────────────
GlobalAveragePool → shape (128,)
Dense(64) + Dropout(0.3) → shape (64,)
Dense(3, softmax) → shape (3,)
─────────────────────────────────────────────────────
[0.72, 0.05, 0.23]  →  predicted: CONFIRMED
```

---

## Every Component Explained

### Conv1D — the core building block

```python
layers.Conv1D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,              # BN handles the bias shift
    kernel_regularizer=l2(1e-4)
)
```

**Filters:** Each filter is a small array of learned weights of size `(kernel_size, n_input_channels)`. With 32 filters, you get 32 different pattern detectors running in parallel. Filter 1 might learn "high SNR + centroid offset = suspicious", Filter 2 might learn something else entirely. The model discovers what patterns matter — you don't tell it.

**Kernel size = 3:** At each position, the filter looks at the current feature and its two neighbours (window of 3). With `padding="same"`, the output has the same length as the input, so no features are lost at the edges.

**use_bias=False:** When BatchNormalization follows immediately, the bias term in Conv1D is redundant — BN's shift parameter (β) does the same job. Removing it saves a few parameters and is standard practice.

**L2 regularization (1e-4):** A small penalty on large weights to prevent overfitting. With only 2,730 training samples, this matters.

---

### BatchNormalization — why training is stable

```python
layers.BatchNormalization()
```

Without BatchNorm, deep networks suffer from what is called internal covariate shift: the distribution of activations changes as training progresses, which makes learning slow and unstable. BatchNorm fixes this by normalizing the activations within each mini-batch to have mean ≈ 0 and variance ≈ 1, then applying learned scale (γ) and shift (β) parameters.

**In practice, BatchNorm:**
- Makes the network much less sensitive to the initial learning rate
- Acts as a mild regularizer
- Allows higher learning rates
- Is placed between Conv1D and ReLU (pre-activation order): Conv → BN → ReLU

---

### ReLU Activation — why not sigmoid or tanh?

```python
layers.Activation("relu")
```

ReLU (Rectified Linear Unit) = `max(0, x)`. Any negative activation becomes 0; positive activations pass through unchanged.

**Why ReLU over sigmoid/tanh:**
- Sigmoid and tanh both saturate — when inputs are large, gradients become near-zero (vanishing gradients). Deep networks learn extremely slowly as a result.
- ReLU doesn't saturate for positive values, so gradients flow cleanly during backpropagation.
- The "dead neuron" problem (ReLU outputs exactly 0 for negative inputs and stays there) is real but manageable in practice — BatchNorm before ReLU reduces it further.

---

### GlobalAveragePooling1D — why not Flatten?

```python
layers.GlobalAveragePooling1D()
```

After three Conv1D blocks, we have a tensor of shape `(37, 128)` — 37 positions, each with 128 feature values.

**Option 1 — Flatten:** Reshape to `(37 × 128,)` = `(4736,)`. Then Dense(64) would need 4,736 × 64 = **303,104 parameters** just in the first Dense layer. With 2,730 training samples, this would massively overfit.

**Option 2 — GlobalAveragePooling1D:** Average across the 37 positions → `(128,)`. Then Dense(64) needs only 128 × 64 = **8,192 parameters**. This is the version we use.

**The tradeoff:** GAP throws away positional information — it can no longer tell whether a pattern appeared at position 5 or position 15. For tabular data where position isn't inherently meaningful, this is acceptable. For raw light curves, position matters more (the transit shape has a specific temporal structure).

**Why it also helps regularization:** GAP effectively averages the activations, which is smoother than taking the raw maximum (as GlobalMaxPool would). This gentle aggregation reduces variance.

---

### Dropout(0.3) — controlled forgetting

```python
layers.Dropout(0.3, seed=42)
```

During training, Dropout randomly sets 30% of the Dense(64) activations to zero at each forward pass. This forces the network to not rely on any single neuron — it must learn redundant representations. At test time, no neurons are dropped (but activations are scaled by 1 - dropout_rate to keep expected values consistent).

**Dropout = an ensemble of 2^64 different networks** (one for each possible mask pattern). Training optimizes across all of them simultaneously, which is a powerful form of regularization.

**Why 0.3, not 0.5?** 0.5 is the classic setting but can be too aggressive when the Dense layer is already small (64 units). 0.3 provides regularization without crippling the layer's capacity.

---

### The Dense head

```python
layers.Dense(64, activation="relu")
layers.Dense(3, activation="softmax")
```

After GAP we have a 128-dimensional feature vector. Dense(64) compresses this further and learns non-linear combinations of the learned CNN features. Dense(3) with softmax produces the final three class probabilities.

**Softmax** ensures outputs are in [0, 1] and sum to exactly 1:
```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```
This is required for multi-class classification.

**Loss function:** Sparse categorical cross-entropy:
```
Loss = − Σ_i y_i × log(ŷ_i)
```
where y_i is the true label (0, 1, or 2) and ŷ_i is the predicted probability for that class. The "sparse" prefix just means we pass integer labels (0/1/2) rather than one-hot vectors — functionally identical.

---

## Parameters

### Architecture config (`config.py`)

```python
BASELINE_CNN_CONFIG = {
    "filters"       : [32, 64, 128],  # Filters per Conv1D block
    "kernel_size"   : 3,              # Window size for each filter
    "dense_units"   : [64],           # Hidden units after GAP
    "dropout_rate"  : 0.3,            # Fraction of Dense neurons dropped during training
    "num_classes"   : 3,              # CONFIRMED / FALSE POSITIVE / CANDIDATE
    "epochs"        : 60,             # Maximum epochs (EarlyStopping fires first)
    "batch_size"    : 32,             # Samples per gradient update
    "learning_rate" : 1e-3,           # Initial Adam learning rate
    "patience"      : 10,             # EarlyStopping: epochs without val_loss improvement
}
```

### Full parameter breakdown

| Parameter | Value | Why |
|-----------|-------|-----|
| `filters` | `[32, 64, 128]` | Doubling filters each block is the standard pyramid — early layers learn simple patterns (32 filters suffice), later layers combine them into complex ones (128 needed). |
| `kernel_size` | `3` | Looks at triplets of adjacent features. Small enough to be local, large enough to capture pairwise + triplet interactions. k=5 would look at quintuplets but doubles parameters. |
| `padding="same"` | — | Output length stays 37 throughout all Conv1D layers. Without this, each Conv1D with k=3 would shrink the sequence by 2, leaving almost nothing for the third block. |
| `dense_units` | `[64]` | One hidden layer with 64 units. Enough capacity to combine the 128 GAP features, without overfitting a 2,730-sample dataset. |
| `dropout_rate` | `0.3` | Moderate regularization on the Dense layer. 0.3 applied to 64 units ≈ 19 neurons dropped per forward pass during training. |
| `batch_size` | `32` | Standard default. Smaller batches (16) give noisier gradients but can generalize better; larger batches (64+) are faster but can converge to sharper, less-generalizable minima. 32 is the sweet spot for this dataset size. |
| `learning_rate` | `1e-3` | Adam's default. ReduceLROnPlateau will halve this automatically if val_loss plateaus. |
| `patience` | `10` | EarlyStopping waits 10 epochs without improvement before halting. With ReduceLROnPlateau (patience=5), the LR will have been halved at least once before EarlyStopping fires. This gives the model a genuine second attempt after each LR reduction. |

### Parameter count breakdown

```
Conv1D(32, k=3):   3 × 1 × 32 = 96 weights
Conv1D(64, k=3):   3 × 32 × 64 = 6,144 weights
Conv1D(128, k=3):  3 × 64 × 128 = 24,576 weights
BatchNorm ×3:      4 params per feature × 3 blocks ≈ 896 params
GAP:               0 params
Dense(64):         128 × 64 + 64 bias = 8,256 params
Dense(3):          64 × 3 + 3 bias = 195 params
─────────────────────────────────────────
Total: 40,163 parameters   (≈ 157 KB)
```

This is deliberately lightweight. Random Forest (300 trees) takes ~150 MB on disk. Our CNN takes 157 KB. The comparison is stark and worth a sentence in the dissertation.

---

## Training Process

### How gradient descent works (briefly)

The model starts with random weights. For each mini-batch of 32 samples:
1. Forward pass: compute predictions, calculate loss
2. Backward pass: compute gradient of loss with respect to every weight (via chain rule / backpropagation)
3. Optimizer step: update weights in the direction that reduces loss

This repeats for every batch in the training set = one **epoch**. With 2,730 training samples and batch_size=32, one epoch = 86 gradient updates.

### Adam optimizer

```python
keras.optimizers.Adam(learning_rate=1e-3)
```

Adam (Adaptive Moment Estimation) is not a fixed step size optimizer. It maintains:
- A running average of past gradients (momentum — helps escape local minima)
- A running average of past squared gradients (adaptive step sizes — large gradients → smaller step, small gradients → larger step)

The result: each weight gets its own effective learning rate, and training is much more stable than plain SGD. It is the default choice for almost all neural network training.

### The three callbacks

**EarlyStopping:**
```python
EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
```
Monitors validation loss after each epoch. If it hasn't improved for 10 consecutive epochs, training stops and weights are restored to the best epoch. This prevents overfitting — the model would otherwise just memorize training noise. In our run: stopped at epoch 58, best was epoch 48.

**ReduceLROnPlateau:**
```python
ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
```
If val_loss doesn't improve for 5 epochs, the learning rate is halved. This gives the model a second wind: when it's stuck at a plateau, a smaller step size can find a narrower valley that the larger step was overshooting. In our run: LR reduced several times from 1e-3 down to ~1.5e-5 before early stopping fired.

**ModelCheckpoint:**
```python
ModelCheckpoint(filepath="baseline_cnn_best.keras", monitor="val_loss", save_best_only=True)
```
Saves the model weights to disk whenever val_loss reaches a new best. This is a safety net — even if the script crashes at epoch 55, we still have the epoch 48 checkpoint.

### Class weights — critical for CANDIDATE recall

```python
from sklearn.utils.class_weight import compute_class_weight
cw_values = compute_class_weight("balanced", classes=[0,1,2], y=y_train)
# → {0: 0.556, 1: 1.201, 2: 2.725}
```

The loss function is reweighted so each CANDIDATE sample costs ~4.9× more than each CONFIRMED sample. Without this, the model learns to ignore CANDIDATE (only 13% of training data) and achieves decent accuracy by predicting mostly CONFIRMED. With it, CANDIDATE F1 goes from ~0.10 to ~0.39.

This is exactly the same approach as `class_weight='balanced'` in scikit-learn. The CNN receives the same treatment as the classical models — a fair comparison.

---

## Results

### Final test set performance (586 samples, never seen during training)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | **85.5%** | 501 / 586 correct |
| **F1 Macro** | **0.754** | Average across all three classes |
| **ROC-AUC** | **0.911** | Strong ranking quality |
| **Cohen's κ** | **0.732** | Solid substantial agreement |
| Best epoch | **48 / 58** | Stopped 10 epochs after best |
| Parameters | **40,163** | Tiny model, fast inference |

### Per-class breakdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| CONFIRMED | 0.88 | 0.90 | **0.89** | 352 |
| FALSE POSITIVE | 0.99 | 0.97 | **0.98** | 163 |
| CANDIDATE | 0.41 | 0.38 | **0.39** | 71 |

### Training dynamics

The val_loss curve tells a clear story:
- Epochs 1–6: rapid learning, LR at 1e-3
- Epoch 6: ReduceLROnPlateau fires first time (LR → 5e-4)
- Epochs 7–14: continued improvement
- Multiple LR reductions follow as the model refines
- Epoch 48: best val_loss = 0.4294
- Epoch 58: EarlyStopping fires (10 epochs without improvement)

The gap between train accuracy (~82%) and val accuracy (~84%) is healthy — no severe overfitting. The BatchNorm + Dropout + L2 regularization worked as intended.

---

## vs Classical ML — The Honest Comparison

| Model | Accuracy | F1 Macro | ROC-AUC | Cohen's κ |
|-------|----------|----------|---------|-----------|
| **Random Forest** | **89.4%** | **0.778** | **0.955** | **0.796** |
| Logistic Regression | 81.9% | 0.763 | 0.924 | 0.692 |
| SVM | 81.9% | 0.739 | 0.897 | 0.681 |
| **Baseline CNN** | **85.5%** | **0.754** | **0.911** | **0.732** |

**Where the CNN sits:** Better than SVM and LR, but 3.9% behind RF on accuracy. This is a meaningful gap on a tabular dataset and is not surprising.

### Why RF beats the Baseline CNN on tabular data

This is well-documented in the literature (Grinsztajn et al., 2022). Key reasons:

1. **Tree-based models handle irregular feature distributions naturally.** Our features span wildly different ranges and distributions — binary flags (0/1), counts, ratios, magnitudes. Trees split on thresholds and are immune to scale. The CNN works on StandardScaled features which partially mitigates this, but the tree still operates directly on the raw feature geometry.

2. **Feature importance asymmetry.** `koi_fpflag_co` has enormous discriminative power. A single decision tree can make this the first split and immediately classify ~60% of the test set correctly. The CNN must learn this implicitly through gradient descent, which is indirect and less efficient for such sharp, discontinuous boundaries.

3. **Data quantity.** 2,730 training samples is a sweet spot where RF (low inductive bias, no assumptions about data structure) consistently outperforms neural networks (high inductive bias, needs data to overcome random initialization). With 100,000 samples, the CNN might close the gap.

4. **The Baseline CNN is a fair but deliberately simple architecture.** It has a single branch with a fixed kernel size of 3. Stage 5 (Genesis CNN) introduces dual branches — one local (k=3) and one global (k=7) — specifically to address this limitation.

### The CNN result is still useful

The Baseline CNN scoring 85.5% vs RF's 89.4% tells us:
- CNNs are competitive on this tabular problem (well above chance and above LR/SVM)
- The single-branch, single-scale architecture leaves performance on the table
- The dual-branch Genesis CNN has a clear hypothesis to test: will capturing both local and global feature interactions close the gap or even exceed RF?

---

## What-if

### Changing `kernel_size`

| k | What changes | Expected effect |
|---|-------------|-----------------|
| 1 | Degenerates to Dense — no local context | Likely worse than Baseline |
| **3** | **Current. Triplet interactions.** | **Best general choice** |
| 5 | Quintuplet interactions, more parameters | Marginal improvement possible, risk of overfitting on 2,730 samples |
| 7 | 7-feature windows | This is what the global branch in Genesis CNN uses — more context per filter |

### Changing `filters`

| filters | Effect |
|---------|--------|
| `[16, 32, 64]` | Smaller model, faster, likely ~1–2% lower accuracy |
| **`[32, 64, 128]`** | **Current. Good capacity for this dataset size.** |
| `[64, 128, 256]` | Larger model, risk of overfitting without more data |

### Removing BatchNormalization

Without BN, training becomes noticeably unstable. Val loss oscillates more and the model typically converges slower. You would likely need to lower the learning rate to 1e-4 to compensate. Expected accuracy drop: ~1–3%.

### Removing class_weight

CANDIDATE F1 would drop from **0.39 → ~0.08**. F1 Macro would drop from **0.754 → ~0.60**. The model would essentially ignore CANDIDATE, predicting CONFIRMED for most ambiguous cases. Never do this on an imbalanced problem without strong justification.

### Using Flatten instead of GlobalAveragePooling1D

Parameters would explode from ~40K to ~340K. With 2,730 training samples, severe overfitting would result — train accuracy near 100%, val accuracy dropping. L2 regularization could partially mitigate this but would require much higher λ.

---

## Limitations

1. **Single receptive field.** Every Conv1D block uses kernel_size=3, so the model only ever looks at triplets of adjacent features. It cannot simultaneously capture both short-range (adjacent feature pairs) and long-range (features far apart in the vector) relationships without going through many layers. The Genesis CNN addresses this with dual kernels.

2. **Feature ordering is arbitrary.** The 37 features were ordered during preprocessing by column order in the original CSV, not by physical or statistical relationship. The CNN's notion of "adjacent" features is therefore constructed, not inherent. This is the most important architectural limitation to acknowledge.

3. **Still loses to Random Forest.** On pre-extracted tabular features, tree-based methods retain their empirical advantage. The CNN adds deep learning complexity without clear performance gains at this scale.

4. **CANDIDATE F1 = 0.39.** Same as with classical models — this is a data problem, not a modelling problem. The CANDIDATE class is administratively labelled (unresolved cases), not physically distinct. No model can reliably separate labels that overlap by definition.

5. **No hyperparameter search.** Filters, kernel size, and dropout were set to principled defaults. A grid search might find marginally better configurations, but is unlikely to close the gap with RF significantly given the sample size constraint.

6. **CPU training only** (in this project). At 40K parameters on 2,730 samples, training takes ~2 minutes on CPU, so GPU is not necessary. TESS-scale applications (millions of TCEs) would require GPU acceleration.

---

## Dissertation Talking Points

### Lead result
> "The Baseline CNN achieves accuracy 85.5%, F1 macro 0.754, and ROC-AUC 0.911 on the held-out test set. This represents a 3.9 percentage point accuracy deficit relative to Random Forest (89.4%), while outperforming both SVM and Logistic Regression."

### Why this result is interesting
> "The Baseline CNN's performance — competitive with but not exceeding classical ML — is consistent with the broader empirical finding that tree-based models retain an advantage on tabular data at small-to-moderate scales (Grinsztajn et al., 2022). Crucially, the Baseline CNN's single-branch, fixed-kernel architecture processes all 37 features at the same spatial scale, which may limit its ability to capture the multi-scale feature interactions that characterise true exoplanet transits."

### Framing for Stage 5
> "The Baseline CNN serves as an architectural control: it demonstrates that Conv1D can be meaningfully applied to pre-extracted KOI features, while exposing the limitation of a single receptive field. The dual-branch Genesis CNN (Stage 5) directly tests whether combining local (kernel=3) and global (kernel=7) processing recovers the performance gap with Random Forest."

### Suggested Methods section sentence
> "A single-branch Conv1D Baseline CNN was constructed with three convolutional blocks (filters: 32, 64, 128; kernel_size=3), followed by GlobalAveragePooling1D and a Dense(64)+Dropout(0.3) head. The model was trained with the Adam optimiser (lr=1e-3), balanced class weights, EarlyStopping (patience=10), and ReduceLROnPlateau (factor=0.5, patience=5) for up to 60 epochs, yielding a total of 40,163 parameters."

### The question your examiners will ask
**"Why use a CNN on tabular data? Isn't this just adding complexity for no reason?"**

The answer has two parts:
1. *Practical:* The CNN is competitive (85.5%) and far from a failed experiment. The complexity is justified by the results.
2. *Architectural:* The Baseline CNN is not the end goal — it is the control condition. The Genesis CNN (dual-branch) tests a specific hypothesis about multi-scale feature interactions. Without the Baseline, you cannot isolate the contribution of the dual-branch design. This is good experimental design.

---

## References

- **LeCun, Y. et al. (1998).** Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. — The foundational CNN paper.
- **Ioffe, S. & Szegedy, C. (2015).** Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*. — The BatchNorm paper.
- **Srivastava, N. et al. (2014).** Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(1), 1929–1958. — The Dropout paper.
- **Kingma, D.P. & Ba, J. (2015).** Adam: A Method for Stochastic Optimization. *ICLR 2015*. — The Adam optimizer paper.
- **Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022).** Why tree-based models still outperform deep learning on tabular data. *NeurIPS 2022*. — Direct empirical support for RF outperforming CNNs on tabular data.
- **Shallue, C.J. & Vanderburg, A. (2018).** Identifying Exoplanets with Deep Learning. *AJ*, 155(2). — The architectural inspiration for this project. Used raw light curves; we adapt the approach to pre-extracted features.

---

*Last updated: Stage 4 — Baseline CNN complete*  
*Next: Stage 5 — Genesis CNN (`src/models/genesis_cnn.py`) — dual-branch Conv1D with local (k=3) + global (k=7) branches*