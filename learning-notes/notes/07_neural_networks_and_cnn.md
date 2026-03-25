# 📚 Notes 07 — Neural Networks & Convolutional Neural Networks (CNNs)

---

## Part 1: Neural Networks From Scratch

### The Brain Analogy

Your brain has ~86 billion neurons. Each neuron receives signals from other neurons, processes them, and sends a signal to other neurons. Complex thoughts emerge from billions of these simple connections.

Artificial neural networks copy this idea — not to be biologically accurate, but because it turns out to be a very powerful mathematical structure.

---

### The Single Neuron (Perceptron)

A single artificial neuron does three things:

```
Step 1: Receive inputs × weights
Step 2: Add them up (+ bias)
Step 3: Apply activation function → output

Input 1 (koi_depth)    ×  weight 1  ─┐
Input 2 (koi_snr)      ×  weight 2  ─┤
Input 3 (fpflag_ss)    ×  weight 3  ─┼─→ SUM (+bias) → Activation → Output
Input 4 (koi_prad)     ×  weight 4  ─┤
...                                  ─┘
```

This is exactly what Logistic Regression does — but with just one neuron.

---

### A Neural Network — Layers of Neurons

Stack many neurons in layers and connect them:

```
INPUT LAYER     HIDDEN LAYER 1    HIDDEN LAYER 2    OUTPUT LAYER
(37 features)   (128 neurons)     (64 neurons)      (3 classes)

  ●                ●                  ●               ● CONFIRMED
  ●   ─────────→   ●   ──────────→   ●   ──────→    ● FALSE POS
  ●                ●                  ●               ● CANDIDATE
  ●                ●                  ●
  ...              ...                ...
  ●                ●                  ●
```

Every neuron in each layer connects to every neuron in the next layer. These are called **fully connected layers** or **dense layers**.

### What Happens at Each Layer?

Each layer learns to detect **increasingly abstract patterns**:

```
Layer 1: Learns simple patterns
  → "Is fpflag_ss = 1?"
  → "Is MES > 10?"
  → "Is planet radius < 20?"

Layer 2: Combines simple patterns
  → "Low MES AND large radius AND fpflag_co = 1"
  → "High MES AND normal stellar density"

Output layer: Final decision
  → "These combined patterns = FALSE POSITIVE"
  → "These combined patterns = CONFIRMED"
```

---

### How Training Works — Backpropagation

This is the magic that makes neural networks learn.

**Forward pass:** Input flows through the network → prediction is made

**Calculate loss:** How wrong was the prediction? (using cross-entropy loss)

**Backward pass (backpropagation):** The error flows **backwards** through the network. Each weight gets a tiny nudge in the direction that reduces the error.

**Gradient descent:** The algorithm that computes which direction to nudge each weight.

```
Training loop (repeats millions of times):

1. Take a batch of 32 stars
2. Forward pass → get predictions
3. Calculate loss (how wrong?)
4. Backpropagate → calculate how to adjust each weight
5. Update weights (tiny step in the right direction)
6. Repeat with next batch
```

> **Analogy:** You're blindfolded on a hilly landscape trying to find the lowest valley (minimum loss). You feel the slope under your feet (gradient) and take a small step downhill. Repeat until you reach the bottom.

---

### Key Neural Network Components

**Activation Functions — Adding Non-Linearity**

Without activation functions, a neural network with 10 layers would be mathematically equivalent to Logistic Regression (one layer). Activation functions add non-linearity — allowing the network to learn complex patterns.

| Activation | Formula | Used Where | Shape |
|---|---|---|---|
| **ReLU** | max(0, x) | Hidden layers | 0 for negative, linear for positive |
| **Sigmoid** | 1/(1+e^-x) | Binary output | S-curve, 0 to 1 |
| **Softmax** | e^x / Σe^x | Multi-class output | All outputs sum to 1 |

**ReLU** (Rectified Linear Unit) is the most common hidden layer activation because it's simple, fast, and avoids the "vanishing gradient" problem.

---

**Dropout — Preventing Overfitting**

Randomly "turns off" a fraction of neurons during each training step.

```
Without dropout:           With dropout (rate=0.4):
All neurons active         40% of neurons randomly disabled
                           per training step

●─●─●─●─●                 ●─ ─●─●─ ─●  (× = disabled)
  (can overfit)              (forces robustness)
```

> **Why does this help?** Neurons can't rely on specific other neurons always being there. They have to learn more general, robust features. Like studying without being allowed to use your notes — forces deeper understanding.

**Dropout is only applied during training.** During testing, all neurons are active (but their weights are scaled proportionally).

---

**Batch Normalisation — Stabilising Training**

After each layer, normalises the activations to have mean≈0 and std≈1.

Benefits:
- Faster training
- More stable gradient flow
- Acts as mild regularisation (reduces overfitting slightly)

Used in your Baseline CNN and tabular Genesis CNN architectures.

---

**Epochs and Batch Size**

| Term | Meaning | Your Project |
|---|---|---|
| **Epoch** | One complete pass through all training data | Up to 60 epochs |
| **Batch** | Subset of data processed together | 32 stars per batch |
| **Iteration** | One update step (processing one batch) | ~2,730/32 ≈ 85 iterations per epoch |

Training for 60 epochs means the model sees each training example up to 60 times.

---

**Early Stopping**

Stop training when the validation loss stops improving — prevents overfitting.

```
Epoch  1: val_loss = 0.85  ← improving
Epoch  5: val_loss = 0.62  ← improving
Epoch 15: val_loss = 0.41  ← improving
Epoch 20: val_loss = 0.39  ← improving
Epoch 25: val_loss = 0.40  ← getting worse!
Epoch 30: val_loss = 0.41  ← still worse
...
Epoch 35: val_loss = 0.42  ← no improvement for 10 epochs

→ STOP. Restore weights from epoch 25 (best val_loss = 0.39)
```

**patience=10** in your project means: wait 10 epochs without improvement before stopping.

---

## Part 2: Convolutional Neural Networks (CNNs)

### Why Not Just Use a Dense Network for Images/Sequences?

If you flatten a 3,197-point light curve into a dense layer of 3,197 inputs connected to 512 neurons — that's **1,636,864 weights** in just the first layer. Massive, slow, and prone to overfitting.

CNNs solve this with a smarter approach: **local pattern detection with weight sharing**.

---

### The Conv1D Layer — A Sliding Window Detector

Instead of connecting every input to every neuron, a Conv1D layer slides a **small window** (kernel) across the input and applies the same set of weights everywhere.

```
Input flux: [93, 83, 20, -26, -39, -124, -135, -96, -79, ...]

Kernel size = 3, Filters = 32:

Window slides along:
Position 1: [93,  83,  20 ] → apply 32 sets of 3 weights → 32 outputs
Position 2: [83,  20, -26 ] → same 32 weight sets → 32 outputs
Position 3: [20, -26, -39 ] → same 32 weight sets → 32 outputs
...
```

**Key insight:** The same kernel is applied at every position. If the network learns to detect a "transit dip pattern" at position 100, it will also detect it at position 500 — **using the same weights**. This is called **weight sharing** and dramatically reduces the number of parameters.

### What Do Different Kernel Sizes Detect?

| Kernel Size | What It Detects | Example |
|---|---|---|
| Small (2-4) | Local, fine-grained patterns | Sharp edges, noise spikes |
| Medium (8-16) | Medium-range patterns | Transit ingress/egress shape |
| Large (32-64) | Long-range patterns | Full transit duration shape |

**In Genesis CNN (light curves):** Kernel starts at 16 (detect wide transit shapes) and decreases (16→8→4→2) as the network gets deeper.

---

### MaxPooling — Downsampling

After convolution, MaxPooling reduces the sequence length by keeping only the maximum value in each window.

```
Before pooling:  [0.2, 0.8, 0.3, 0.9, 0.1, 0.7]
MaxPool(size=2): [     0.8,       0.9,       0.7]
                  ↑max of          ↑max of      ↑max of
                  [0.2,0.8]       [0.3,0.9]   [0.1,0.7]
```

Benefits:
- Reduces sequence length (faster computation)
- Keeps the strongest activations (most important patterns)
- Makes the network slightly position-invariant (transit dip in different part of curve is still detected)

---

### GlobalAveragePooling — Final Compression

After multiple conv+pool layers, the sequence is compressed to a single vector by averaging across all positions.

```
Shape before GAP: (batch, 12, 128)  ← 12 time steps, 128 filters
Shape after GAP:  (batch, 128)       ← one 128-dimensional vector per star
```

This single vector is then fed into dense layers for final classification.

Used in your **tabular Genesis CNN**.

---

### The Full CNN Architecture Flow

```
Input Sequence (3197 time steps × 1 channel)
         ↓
Conv1D (learn local patterns) → ReLU → MaxPool (compress)
         ↓
Conv1D (learn higher patterns) → ReLU → MaxPool (compress)
         ↓
Conv1D (learn even higher patterns) → ReLU → MaxPool (compress)
         ↓
Conv1D (final patterns) → ReLU → MaxPool (compress)
         ↓
Flatten or GlobalAveragePooling (→ 1D vector)
         ↓
Dense Layer (combine all learned patterns)
         ↓
Dropout (prevent overfitting)
         ↓
Output Layer (final prediction)
```

---

## Part 3: Your Baseline CNN (Stage 4)

The Baseline CNN is a simple, single-branch CNN used as a **deep learning reference point** before the Genesis CNN.

### Architecture

```python
Input (37 features, 1 channel)
    │
Conv1D(32 filters, kernel=3) + BatchNorm + ReLU
    │
Conv1D(64 filters, kernel=3) + BatchNorm + ReLU
    │
Conv1D(128 filters, kernel=3) + BatchNorm + ReLU
    │
GlobalAveragePooling1D → (128,)
    │
Dense(64, ReLU) + Dropout(0.3)
    │
Dense(3, Softmax) → 3 Classes
```

### Key Design Choices

| Choice | Reason |
|---|---|
| kernel=3 throughout | Captures local adjacent feature interactions only |
| 3 conv layers | Progressively deeper pattern detection |
| GlobalAvgPool | Compact summary without massive parameter count |
| BatchNorm after each conv | Faster, more stable training |
| Dropout(0.3) | Prevents overfitting on small dataset |

### Result: 85.5% accuracy
This is the benchmark the Genesis CNN must beat. It's also ~3.9% below Random Forest — motivating the more sophisticated Genesis architecture.

---

## Key Concepts Summary

```
Neuron
    └── Takes inputs × weights, sums them, applies activation

Layers
    └── Input → Hidden (1 or more) → Output

Backpropagation
    └── How errors flow backward to adjust weights

Gradient Descent
    └── Algorithm that adjusts weights to minimise loss

Activation Function (ReLU)
    └── Adds non-linearity — enables learning complex patterns

Dropout
    └── Randomly disables neurons during training — prevents overfitting

Batch Normalisation
    └── Normalises layer outputs — speeds up training

Conv1D
    └── Sliding window detector — weight sharing, fewer parameters

Kernel Size
    └── Width of the sliding window — larger = wider pattern detection

MaxPooling
    └── Keeps maximum values — compresses sequence, adds invariance

GlobalAveragePooling
    └── Averages across all positions → 1D vector

Early Stopping
    └── Stops training when validation loss stops improving
```

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **Neuron** | Basic unit — receives inputs, applies weights, outputs signal |
| **Layer** | Group of neurons processing at the same level |
| **Dense Layer** | Every neuron connects to every neuron in next layer |
| **Backpropagation** | Algorithm for computing how to adjust weights |
| **Gradient Descent** | Optimisation — adjusts weights to reduce loss |
| **ReLU** | Activation function: max(0,x) — standard for hidden layers |
| **Softmax** | Converts scores to probabilities summing to 1 (multi-class output) |
| **Dropout** | Randomly disables neurons during training — reduces overfitting |
| **Batch Normalisation** | Normalises layer outputs for stable training |
| **Epoch** | One complete pass through all training data |
| **Batch** | Subset of data processed in one training step |
| **Early Stopping** | Stops training when validation stops improving |
| **Conv1D** | 1D convolutional layer — sliding window over sequence |
| **Kernel** | The sliding window in convolution — its weights are learned |
| **MaxPooling** | Keeps maximum value in each window — compresses sequence |
| **GlobalAveragePooling** | Averages all positions → single vector per sample |
| **Weight Sharing** | Same kernel weights used at every position — key CNN advantage |

---

*Notes from MSc Data Science Dissertation — Exoplanet Candidate Vetting*
