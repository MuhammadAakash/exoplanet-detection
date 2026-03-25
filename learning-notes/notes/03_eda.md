# 📚 Notes 03 — Exploratory Data Analysis (EDA)

---

## What is EDA?

EDA = **Exploratory Data Analysis**

It's what you do *before* building any model — understanding your data so there are no surprises later.

> **Analogy:** A doctor examining a patient before surgery. You don't just cut straight in — you check vitals, run tests, understand what you're working with.

**The goal:** Understand your data's shape, spot problems early, and make smarter modelling decisions.

---

## The 5 Things EDA Always Checks

```
1. What does the data look like?       → Shape, types, column names
2. Are there missing values?           → Gaps in data
3. What is the class distribution?     → Imbalance problems
4. What do feature distributions look like? → Skew, outliers, shape
5. Are features correlated?            → Redundancy between features
```

---

## Check 1 — What Does the Data Look Like?

Basic questions first:
- How many rows (examples) and columns (features)?
- What are the data types — numbers, text, categories?
- Does anything look obviously wrong?

**In your project:**
```
koi_data.csv → 3,901 rows × 141 columns (raw)
After cleaning → 3,900 rows × 37 features + 1 label
```

---

## Check 2 — Missing Values

Missing values = gaps in your data. Measurements never taken or corrupted.

> **Analogy:** A patient's medical record where the blood pressure column is blank for some visits.

### In Your Project
- **20 columns completely empty** — orbital eccentricity and others that weren't computed for most KOIs → dropped entirely
- **Remaining missing values** → handled with **Median Imputation**

### What is Median Imputation?
Fill the missing value with the **median** (middle value) of that feature from the training set.

```python
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)       # ← learns the median from TRAINING DATA ONLY
X_train = imputer.transform(X_train)
X_val   = imputer.transform(X_val)    # ← uses training median on val
X_test  = imputer.transform(X_test)   # ← uses training median on test
```

> ⚠️ **Critical rule:** Fit the imputer on TRAINING DATA ONLY. Never use validation or test data to calculate the median. This would be data leakage.

### Why Median and Not Mean?

| Statistic | Problem | Example |
|---|---|---|
| **Mean** | Pulled by extreme outliers | Star radii: [0.8, 0.9, 0.9, 1.0, 1.1, **47.0**] → Mean = 8.6 (wrong!) |
| **Median** | Ignores outliers | Same data → Median = 0.95 ✅ |

---

## Check 3 — Class Distribution

How many examples does each class have?

### Tabular Pipeline (koi_data.csv)
```
CONFIRMED      → 2,293 stars  (58.8%)  ← majority
FALSE POSITIVE → 1,008 stars  (25.8%)
CANDIDATE      →   592 stars  (15.2%)  ← minority ⚠️
```

### Light Curve Pipeline (exoTrain.csv)
```
Non-Planet → 5,050 stars  (99.3%)  ← overwhelming majority
Planet     →    37 stars   (0.7%)  ← severe minority ⚠️⚠️
                                      136:1 ratio!
```

### Why Imbalance is a Problem

A model that **always predicts the majority class** gets:
- Tabular: 58.8% accuracy (doing nothing useful)
- Light curves: 99.3% accuracy (finding zero planets!)

Both look decent on paper. Both are completely useless.

### How to Fix It: `class_weight='balanced'`

This tells the model to pay more attention to rare classes during training.

**How it works:**
```
weight = total_samples / (n_classes × n_samples_in_class)

Tabular (CANDIDATE): weight = 3900 / (3 × 592) = 2.19×
Light curve (Planet): weight = 5087 / (2 × 37)  = 68.7×
```

> **Analogy:** Giving the teacher double marks for questions about the rare topic — so students study it more carefully.

---

## Check 4 — Feature Distributions

For each feature, plot a **histogram** — a bar chart showing how many examples have each value range.

### Types of Distributions

| Shape | Name | What It Means | Example Feature |
|---|---|---|---|
| Bell curve | Normal | Symmetric, most values in middle | `koi_steff` (temperature) |
| Long right tail | Right-skewed | Most values small, few very large | `koi_max_mult_ev` (MES) |
| Two peaks | Bimodal | Natural separation in data | Sometimes seen in `koi_depth` |
| Only 0 or 1 | Binary | Flag-type feature | `koi_fpflag_ss` |

### Why Does Distribution Shape Matter?

**StandardScaler** assumes roughly normal data. It works by:
```
scaled_value = (value - mean) / standard_deviation
```
This works well for normal distributions. For heavily skewed data, the scaler still helps but may not fully solve the problem.

> ⚠️ **Same rule as imputer:** Fit scaler on training data only. Apply to val and test.

---

## Check 5 — Feature Correlation

Correlation = how much two features move together.

| Correlation Value | Meaning |
|---|---|
| +1.0 | Perfect positive — one goes up, other goes up exactly |
| +0.7 | Strong positive — usually move together |
| 0.0 | No relationship |
| −0.7 | Strong negative — one goes up, other goes down |
| −1.0 | Perfect negative |

### Why Does Correlation Matter?

**Highly correlated features are redundant.** They carry the same information twice.

**Example in your project:**
- `koi_prad` (planet radius in Earth radii) and `koi_ror` (radius ratio) are highly correlated
- `koi_ror` is literally used to calculate `koi_prad` — so they say the same thing in different units

### Which Models Care About Correlation?

| Model | Handles Correlation? |
|---|---|
| Random Forest | ✅ Yes — handles naturally |
| SVM | ✅ Mostly fine |
| Logistic Regression | ❌ Struggles — correlated features cause unstable coefficients |
| CNN | ✅ Yes — learns to handle it |

---

## EDA for Light Curve Data

Different from tabular EDA because input is a **time series** (3,197 flux measurements per star).

### What to Look For Visually

**Planet star light curve:**
```
Flux
  │  ████████████████████████████
  │  ████████████████████████████   ← flat baseline
  │  ████████              ███████
  │  ████      ████████████
  │                                 ← transit dip (planet crossing)
  └──────────────────────────────── Time
```

**Non-planet star light curve:**
```
Flux
  │  ███   ██████         ████████
  │  ████        ████████████████   ← irregular fluctuations
  │                                    no consistent dip
  └──────────────────────────────── Time
```

### After Normalisation (median=0, min=−1)

**Before:** flux values range from −300,000 to +1,400,000 (huge range!)
**After:** all curves compressed to roughly [−1, +0.5] range

The transit dip always goes to **exactly −1** at its deepest point. Every star is on the same scale. The CNN can now compare shapes without being distracted by how bright the star is.

---

## Key EDA Findings in This Project

### Tabular Pipeline
| Finding | Implication for Modelling |
|---|---|
| 20 columns completely empty | Drop them |
| 15% CANDIDATE minority | Use `class_weight='balanced'` |
| FP flags are near-binary (0 or 1) | Will be top features in Random Forest |
| MES and SNR are right-skewed | StandardScaler still appropriate |
| Transit depth spans 4 orders of magnitude | Wide range — model needs to handle this |
| Highly correlated pairs exist | Random Forest handles naturally; LR may struggle |

### Light Curve Pipeline
| Finding | Implication for Modelling |
|---|---|
| 0.7% planet minority (136:1) | Monitor ROC-AUC not accuracy |
| Flux range: −300,000 to +1,400,000 | Normalisation is essential |
| 0 missing values | No imputation needed |
| Planet stars have larger flux variance | Statistical feature `flux_std` will be discriminative |
| Transit dips visible in planet stars | Normalised shape is the key CNN input |

---

## EDA Figures in Your Project

### Tabular Pipeline (`run_eda.py`)
*(figures saved to `results/figures/eda_*.png`)*

| Figure | What It Shows |
|---|---|
| Class distribution bar/pie | Motivates class weighting |
| Feature distributions | Shows skew, outliers, scale |
| Correlation heatmap | Identifies redundant feature pairs |
| Missing value summary | Shows which columns were dropped |

### Light Curve Pipeline (`eda_lc.py`)
*(figures saved to `results/figures/lc_eda_*.png`)*

| Figure | What It Shows |
|---|---|
| `lc_eda_01_class_distribution` | 136:1 imbalance — motivates AUC monitoring |
| `lc_eda_02_sample_light_curves` | Raw flux for planet vs non-planet stars |
| `lc_eda_03_flux_statistics` | Std, mean, min flux per class |
| `lc_eda_04_normalised_examples` | Same curves after normalisation — CNN-ready |

---

## EDA Checklist (Use Before Every Project)

```
☐ Check shape (rows × columns)
☐ Check data types for each column
☐ Check for missing values — how many? Where?
☐ Check class distribution — is it balanced?
☐ Plot distributions for key features
☐ Look for outliers (extreme values)
☐ Check feature correlations
☐ Look at a few raw examples — does the data look sensible?
☐ Verify labels match expectations
☐ Check for duplicate rows
```

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **EDA** | Exploratory Data Analysis — understanding data before modelling |
| **Distribution** | How values of a feature are spread across examples |
| **Class Imbalance** | When one class has far fewer examples than others |
| **Median Imputation** | Filling missing values with the median of training data |
| **Data Leakage** | Using test/validation info during training — invalidates results |
| **Correlation** | How much two features move together |
| **Normalisation** | Rescaling values to a fixed range |
| **Standardisation** | Rescaling so mean=0, std=1 (StandardScaler) |
| **Histogram** | Bar chart showing the distribution of one feature |
| **Outlier** | An extreme value far from most of the data |
| **class_weight='balanced'** | Tells model to weight rare classes more during training |

---

*Notes from MSc Data Science Dissertation — Exoplanet Candidate Vetting*
