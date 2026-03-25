# 📚 Dissertation Study Notes
## Exoplanet Candidate Vetting Using a Simplified Genesis CNN
### MSc Data Science — Mohammad Aakash — University of Hertfordshire

---

## How to Use These Notes

These notes are structured to build understanding from the ground up.
Read them in order for the first time. Use them as reference later.

Each note file covers one complete topic with:
- Simple explanations and analogies
- Technical details
- How it applies to YOUR project specifically
- Key terms glossary

---

## 📂 Notes Index

| File | Topic | What You'll Learn |
|---|---|---|
| `01_what_is_machine_learning.md` | Machine Learning Basics | What ML is, supervised learning, train/val/test split |
| `02_data_and_features.md` | Data & Features | Your 37 features explained one by one |
| `03_eda.md` | Exploratory Data Analysis | How to understand data before modelling |
| `04_random_forest.md` | Random Forest | Decision trees → ensemble → voting |
| `05_svm.md` | Support Vector Machine | Maximum margin, kernels, C parameter |
| `06_logistic_regression.md` | Logistic Regression | Weights, sigmoid, linear decision boundary |
| `07_neural_networks_and_cnn.md` | Neural Networks & CNNs | Neurons, backprop, Conv1D, pooling |
| `08_genesis_cnn.md` | Genesis CNN | Your main model — both implementations |

---

## 🗺️ The Learning Path

```
Machine Learning Basics (Note 01)
        ↓
Data & Features (Note 02) ← understand your 37 features
        ↓
EDA (Note 03) ← understand your data visually
        ↓
Classical Models:
  Random Forest (Note 04) ← best classical model (89.4%)
  SVM (Note 05)           ← maximum margin (86.1%)
  Logistic Regression (Note 06) ← linear baseline (80.3%)
        ↓
Deep Learning:
  Neural Networks & CNNs (Note 07) ← how they work
  Genesis CNN (Note 08)            ← your main contribution (~91%)
```

---

## 🎯 Your Project at a Glance

### Pipeline 1 — Tabular Features
```
Input:    koi_data.csv — 3,900 stars × 37 features
Task:     3-class classification
          CONFIRMED / FALSE POSITIVE / CANDIDATE
Models:   Random Forest (89.4%) | SVM (86.1%) | Logistic Regression (80.3%)
          Baseline CNN (85.5%)  | Genesis CNN (~91%) ← BEST
Key idea: Multi-scale dual-branch CNN on pre-extracted astrophysical features
```

### Pipeline 2 — Light Curve Flux
```
Input:    exoTrain.csv — 5,087 stars × 3,197 flux measurements
Task:     Binary classification — Planet / Non-Planet
Models:   Random Forest | SVM | Logistic Regression | Genesis CNN (~95%+)
Key idea: Single-branch sparse Conv1D on raw Kepler flux (Visser et al. 2022)
Imbalance: 136:1 (37 planets vs 5,050 non-planets)
```

---

## 📖 Quick Concept Reference

### The Models — One Line Each

| Model | One-Line Explanation |
|---|---|
| Random Forest | 300 decision trees vote on the answer |
| SVM | Finds the widest possible gap between classes |
| Logistic Regression | Weighted sum of features → probability |
| Baseline CNN | Single-branch Conv1D on tabular features |
| Genesis CNN (tabular) | Dual-branch Conv1D — local (k=3) + global (k=7) kernels |
| Genesis CNN (light curves) | Single-branch Conv1D on raw flux — Visser et al. 2022 |

### The Metrics — One Line Each

| Metric | One-Line Explanation |
|---|---|
| Accuracy | % of predictions that were correct |
| F1 Macro | Average F1 across all classes — penalises ignoring minorities |
| ROC-AUC | How well model ranks positives above negatives (1.0 = perfect) |
| Cohen's κ | Accuracy adjusted for chance — more honest |
| MCC | Best single metric for imbalanced data |
| F1-Planet | Specifically how well planets are found |
| Recall | Of all real planets, how many did the model find? |

### The Key Terms — One Line Each

| Term | One-Line Explanation |
|---|---|
| Feature | A measurable property used as model input |
| Label | The correct answer the model predicts |
| Overfitting | Memorising training data — fails on new data |
| Data leakage | Cheating — using test/answer info during training |
| Class imbalance | One class has far fewer examples |
| class_weight='balanced' | Makes rare classes count more during training |
| StandardScaler | Transforms features to mean=0, std=1 |
| Median imputation | Fills missing values with training set median |
| Stratified split | Preserves class ratios in each data split |
| Backpropagation | How neural networks learn — error flows backward |
| Gradient descent | Algorithm for adjusting weights to reduce loss |
| Dropout | Randomly disables neurons — prevents overfitting |
| Batch normalisation | Normalises layer outputs — faster, stable training |
| Conv1D | Sliding window over sequence — detects local patterns |
| Kernel | The sliding window in convolution |
| Weight sharing | Same kernel used everywhere — key CNN advantage |
| MaxPooling | Keeps maximum values — compresses sequence |
| EarlyStopping | Stops training when validation stops improving |

---

## 🔑 The Most Important Concepts for Your Viva

1. **Why class weights?** — CANDIDATE (15%) and Planet (0.7%) are minorities. Without weights, model ignores them.

2. **Why two CNN branches?** — Local branch (k=3) captures adjacent feature patterns. Global branch (k=7) captures wider cross-group patterns. Together = richer representation than either alone.

3. **Why ROC-AUC for light curves?** — 136:1 imbalance means accuracy is meaningless. ROC-AUC measures genuine discrimination regardless of threshold.

4. **Why StandardScaler before SVM/LR but not RF?** — SVM and LR measure distances/weights and are scale-sensitive. RF uses rank-based splits and doesn't care about scale.

5. **What makes Genesis "simplified"?** — >95% fewer parameters than AstroNet. Single view instead of dual view. Sparse architecture generalises better on small datasets.

6. **Why not use ALL 141 features?** — 20 empty, 17 identifiers, 2 leaky. Irrelevant features add noise and confuse the model.

7. **Why train/val/test split?** — Test set must be unseen for honest evaluation. Validation set tunes model without contaminating the test.

8. **What is the CANDIDATE class challenge?** — CANDIDATE means "uncertain" — even human experts disagree. It's inherently the hardest class. All 5 models struggle most here.

---

*Generated as part of MSc Data Science Dissertation — Exoplanet Candidate Vetting*
*University of Hertfordshire | NASA Exoplanet Archive | Kepler Mission*
