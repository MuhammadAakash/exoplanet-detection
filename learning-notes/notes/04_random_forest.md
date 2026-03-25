# 📚 Notes 04 — Random Forest

---

## Step 1: The Building Block — Decision Trees

Before Random Forest, understand its building block: the **Decision Tree**.

### How a Decision Tree Thinks

A decision tree asks a series of yes/no questions to reach a decision.

```
Real-life example — Should I take an umbrella?

Is it cloudy?
    ├── NO  → Don't take umbrella ☀️
    └── YES → Is rain forecast?
                  ├── NO  → Maybe leave it 🌥️
                  └── YES → Take umbrella ☔
```

### Decision Tree on Your Exoplanet Data

```
Is koi_fpflag_ss = 1?  (secondary eclipse detected?)
    │
    ├── YES → FALSE POSITIVE (almost certainly)
    │
    └── NO  → Is koi_max_mult_ev > 15?  (strong repeating signal?)
                  │
                  ├── YES → Is koi_prad < 20?  (planet-sized?)
                  │           ├── YES → CONFIRMED ✅
                  │           └── NO  → FALSE POSITIVE ❌
                  │
                  └── NO  → CANDIDATE ❓
```

**The machine figures out these questions automatically from data.** It tries every possible split on every feature and picks the one that best separates the classes.

### The Problem With One Tree

A single decision tree:
- Makes rigid, brittle decisions
- Easily **overfits** — memorises training data, fails on new data
- Like asking just one expert who might be having a bad day

---

## Step 2: Random Forest — Many Trees, One Answer

### The Core Idea

Grow **hundreds of decision trees** and let them vote on the final answer.

> **Analogy:** Instead of asking one expert, ask 300 different experts. Each gives their opinion. Go with the majority.

This is called an **ensemble method** — combining many weak models into one strong model.

### The Two Sources of Randomness

**Randomness #1 — Bootstrap Sampling (Random Rows)**

Each tree trains on a *random sample* of training data, drawn *with replacement*.

```
Training data: Stars 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

Tree #1 sees: Stars 1, 1, 3, 5, 5, 6, 8, 9, 9, 10  ← some repeated, some missing
Tree #2 sees: Stars 2, 2, 3, 4, 6, 7, 7, 8, 9, 10  ← different sample
Tree #3 sees: Stars 1, 2, 4, 4, 5, 6, 7, 8, 10, 10 ← another different sample
```

Each tree sees a slightly different version of the world.

**Randomness #2 — Feature Subsampling (Random Columns)**

At each split point, each tree only considers a **random subset of features**.

```
All 37 features available

Tree #1 at split 1: Can only choose from features {3, 7, 12, 19, 28}
Tree #1 at split 2: Can only choose from features {1, 5, 18, 22, 36}
Tree #2 at split 1: Can only choose from features {4, 8, 11, 25, 33}
```

This forces **diversity** — trees can't all copy the same obvious feature.

### Making a Prediction

```
New star arrives → fed into all 300 trees simultaneously

Tree 1   says: CONFIRMED
Tree 2   says: CONFIRMED
Tree 3   says: FALSE POSITIVE
Tree 4   says: CONFIRMED
...
Tree 300 says: CONFIRMED

240 trees say CONFIRMED
 45 trees say FALSE POSITIVE
 15 trees say CANDIDATE

Final prediction: CONFIRMED  (240/300 = 80% confidence)
```

---

## Feature Importance — A Free Bonus

Random Forest tells you **which features were most useful** across all 300 trees.

### How It's Calculated (Gini Importance)
Every time a tree uses a feature to make a split, it measures how much that split improved the predictions. Sum this improvement across all trees for each feature. Normalise to sum to 1.

### Expected Top Features in Your Project

| Rank | Feature | Why It's Important |
|---|---|---|
| 1 | `koi_fpflag_ss` | Secondary eclipse = almost certain false positive |
| 2 | `koi_max_mult_ev` | MES = measures consistency of repeating signal |
| 3 | `koi_fpflag_co` | Centroid offset = signal from background star |
| 4 | `koi_model_snr` | Signal-to-noise = how clear the detection is |
| 5 | `koi_fpflag_nt` | Not transit-like = signal doesn't look like a planet |

These match what NASA's Robovetter uses — scientifically validating your model's behaviour.

---

## Why Random Forest Works So Well

| Challenge | How Random Forest Handles It |
|---|---|
| Missing values | Trees can split around them naturally |
| Outliers (extreme values) | Tree splits based on rank order, not actual value |
| Correlated features | Feature subsampling forces trees to explore alternatives |
| Feature scaling | Not needed — trees don't care about absolute scale |
| Overfitting | Voting across 300 trees prevents any one bad tree dominating |

> **No scaling needed!** Unlike SVM and neural networks, Random Forest doesn't require StandardScaler. It would make no difference.

---

## Your Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=300,       # 300 trees in the forest
    max_depth=None,         # Trees grow as deep as needed
    min_samples_split=2,    # Minimum 2 samples to make a split
    class_weight="balanced",# Pay more attention to rare classes
    random_state=42,        # Fixed seed for reproducibility
    n_jobs=-1,              # Use all CPU cores for speed
)
```

**Why 300 trees?** More trees = more stable predictions, up to a point. Beyond ~300, performance barely improves but training time keeps increasing.

**Why `class_weight="balanced"`?** Without it, the model ignores the CANDIDATE class (only 15% of data). With it, each CANDIDATE example is weighted ~2.2× more.

---

## Results in Your Project

| Metric | Random Forest | Why This Result Makes Sense |
|---|---|---|
| Accuracy | 89.4% | Best classical model — handles tabular data excellently |
| F1 Macro | 0.821 | Good across all 3 classes, including CANDIDATE |
| ROC-AUC | 0.962 | Excellent class discrimination |
| Cohen's κ | 0.828 | Strong agreement beyond chance |

**Random Forest is the best classical model** and sets the target for the Genesis CNN to beat.

---

## Random Forest vs Decision Tree

| | Decision Tree | Random Forest |
|---|---|---|
| Number of models | 1 | 300+ |
| Overfitting risk | HIGH | LOW |
| Variance | HIGH (sensitive to small data changes) | LOW (averaging reduces variance) |
| Speed | Very fast | Slower (but parallelisable) |
| Interpretability | Easy (you can draw it) | Harder (300 trees) |
| Performance | Good | Better |

---

## Key Concepts Summary

```
Decision Tree
    └── Asks yes/no questions to classify examples

Bootstrap Sampling
    └── Each tree trained on a random sample (with replacement)

Feature Subsampling
    └── Each split only sees a random subset of features

Ensemble Method
    └── Combining many models into one stronger model

Majority Vote
    └── Final prediction = class most trees agree on

Feature Importance
    └── Which features were most useful across all trees

Overfitting
    └── Memorising training data — Random Forest resists this via voting
```

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **Decision Tree** | Model that makes decisions through a series of yes/no questions |
| **Random Forest** | Ensemble of many decision trees with voting |
| **Ensemble Method** | Combining many models to make better predictions than any single model |
| **Bootstrap Sampling** | Random sampling with replacement — each tree gets different data |
| **Feature Subsampling** | Each split only considers a random subset of features |
| **Feature Importance** | Score showing how useful each feature was across all trees |
| **Gini Importance** | The specific way Random Forest measures feature usefulness |
| **Overfitting** | Model memorises training data, fails on new data |
| **n_estimators** | Number of trees in the forest (your project: 300) |
| **class_weight='balanced'** | Automatically weights rare classes more during training |

---

*Notes from MSc Data Science Dissertation — Exoplanet Candidate Vetting*
