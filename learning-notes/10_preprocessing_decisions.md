# 10 — Preprocessing Decisions and Why

> Every preprocessing choice has a reason. This file documents each decision, the alternative that was considered, and why the chosen approach is correct for this specific dataset.

---

## The Preprocessing Pipeline — Full Overview

```
Raw NASA KOI CSV (3,901 rows, 141 columns)
              ↓
Step 1: Drop identifier and administrative columns
              ↓
Step 2: Drop the leaky target-related column (koi_pdisposition)
              ↓
Step 3: Drop columns excluded as leaky features (koi_score)
              ↓
Step 4: Drop columns with > 50% missing values
              ↓
Step 5: Add 3 engineered features
              ↓
Step 6: Encode target labels (koi_disposition → integer)
              ↓
Step 7: Stratified train/val/test split (70/15/15)
              ↓
Step 8: Fit median imputer on training data
              ↓
Step 9: Transform all splits with fitted imputer
              ↓
Step 10: Fit StandardScaler on training data
              ↓
Step 11: Transform all splits with fitted scaler
              ↓
Step 12: Save imputer, scaler, and splits to disk
```

---

## Decision 1 — Which Columns to Drop

### Columns dropped: identifier and administrative

```python
DROP_COLS_ADMIN = [
    'rowid', 'kepid', 'kepoi_name', 'kepler_name',  # Identifiers
    'koi_vet_stat', 'koi_vet_date',                  # Administrative
    'koi_disp_prov', 'koi_comment',                  # Text fields
    'koi_tce_plnt_num', 'koi_tce_delivname',         # Pipeline references
    'koi_quarters', 'koi_trans_mod',                 # Qualitative metadata
    'koi_datalink_dvr', 'koi_datalink_dvs',          # URL links
    'koi_sparprov', 'koi_fittype',                   # Method identifiers
    'koi_parm_prov', 'koi_limbdark_mod',             # Model references
]
```

**Why:** These are administrative labels, not physical measurements. A planet's kepid (Kepler target identifier) is an arbitrary number with no physical meaning. Including identifiers would cause the model to memorise which specific KOIs are planets — a form of look-up rather than generalisation.

---

### Columns dropped: leaky target-related

```python
DROP_COLS_LEAKY = [
    'koi_pdisposition',  # Preliminary disposition — derived from same process as target
    'koi_score',         # Robovetter score — used to generate disposition labels
]
```

**What is `koi_pdisposition`?**  
The automated preliminary disposition, set before human review. It is either CANDIDATE or FALSE POSITIVE — the automated system's first guess before human astronomers reviewed the evidence. Since the final `koi_disposition` was set in part by reviewing this preliminary disposition, including it would mean the model is predicting the final answer using the preliminary answer — trivially easy and scientifically meaningless.

**What is `koi_score`?**  
The Robovetter's confidence score (0.0–1.0). As explained in the EDA, this was computed by the same pipeline that generated the labels. Including it would be pure label leakage.

**Why "leakage" is the right word:**  
Leakage means including information in training that would not be available in real-world deployment. If I am building a system to vet new exoplanet candidates (for TESS, for example), I would not have the Robovetter score or the preliminary human disposition for those new candidates. They are unavailable at prediction time. Including them in training creates a model that cannot generalise to deployment.

---

### Columns dropped: > 50% missing

```python
DROP_THRESHOLD = 0.50

# Identified in EDA — columns with > 50% empty values
DROP_COLS_MISSING = [
    'koi_eccen', 'koi_eccen_err1', 'koi_eccen_err2',    # Assumed circular
    'koi_longp', 'koi_longp_err1', 'koi_longp_err2',    # Assumed circular
    'koi_ingress', 'koi_ingress_err1', 'koi_ingress_err2', # Model-dependent
    'koi_sage', 'koi_sage_err1', 'koi_sage_err2',        # Rarely measured
    # ... and others reaching > 50% empty
]
```

**Why 50% as the threshold:**  
If more than 50% of a column is missing, then imputing it means inventing more data than you have measured data. The imputed values are not observations — they are placeholders generated from the distribution of the 50% that do exist. At some point, the invented data is more noise than signal.

50% is a reasonable and commonly used threshold. It is not arbitrary — it is the point where invented exceeds measured.

**Alternative considered: 30% threshold**  
Would have kept more columns but would have meant ~70% of some columns are invented. Rejected.

**Alternative considered: 80% threshold**  
Would have kept fewer columns, losing some with moderate missingness that still carry genuine information. Rejected.

---

### Error columns not used as features

```python
# Columns with _err1, _err2 suffix are measurement uncertainties
# e.g., koi_period_err1, koi_depth_err2
# These are NOT included as model features
```

**Why exclude uncertainty columns:**  
Including measurement uncertainties would help a model that can integrate value + uncertainty as Bayesian estimates. The models used here (RF, SVM, CNN) do not have a mechanism to naturally incorporate paired (value, uncertainty) information. The uncertainties would just add noise to the feature vector. They could be incorporated in future work using uncertainty-aware architectures.

---

## Decision 2 — Label Encoding

```python
LABEL_MAP = {
    'CONFIRMED':      0,
    'FALSE POSITIVE': 1,
    'CANDIDATE':      2
}

df['target'] = df['koi_disposition'].map(LABEL_MAP)
```

**Why integer encoding:**  
Sklearn classifiers and Keras require integer targets. The specific numbers (0, 1, 2) are arbitrary — the model does not treat 0 as "better" than 2. Any three distinct integers would produce identical results.

**Alternative considered: one-hot encoding**  
Would represent each class as a binary vector: CONFIRMED → [1,0,0], FP → [0,1,0], CANDIDATE → [0,0,1]. Required for Keras softmax output (handled inside the model, not in preprocessing). For sklearn models, integer encoding is standard.

---

## Decision 3 — Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

# Step 1: Separate test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.15,       # 15% for test
    stratify=y,           # Preserve class ratios
    random_state=42       # Reproducibility
)

# Step 2: Separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.1765,     # 15% of total (15% / 85% ≈ 0.1765)
    stratify=y_train_val,
    random_state=42
)
```

**Result:**
```
Training:   2,730 rows (70%)  — model learning
Validation:   585 rows (15%)  — hyperparameter tuning, early stopping
Test:         586 rows (15%)  — final evaluation (touched ONCE)
```

**Why 70/15/15:**  
Standard split for a dataset of this size (3,901 rows). Provides enough training data for reliable learning while keeping separate val and test sets for unbiased tuning and evaluation.

**Why stratified:**  
Without stratification, random chance could produce a test set with disproportionately few CANDIDATE examples (only 477 total). With only ~71 candidates expected in a 15% test set, variance would be high. Stratification guarantees exactly the right proportion in each split.

**Why separate val from test:**  
The validation set is used during training for:
- Early stopping (stop when val loss stops improving)
- Hyperparameter search (which learning rate gives best val F1?)
- Architecture decisions (which kernel size works best?)

If I used the test set for these decisions, the test set would be contaminated — I would have effectively trained on it. The golden rule: **test set is touched exactly once, after all decisions are final.**

**Why random_state=42:**  
Reproducibility. Any fixed integer works. Using the same seed means you can run the preprocessing again and get exactly the same split, enabling reproducible results.

---

## Decision 4 — Missing Value Imputation

```python
from sklearn.impute import SimpleImputer

# CRITICAL: Fit on training data ONLY
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)    ← compute medians from training set only

# Apply (transform) to all sets
X_train_imp = imputer.transform(X_train)
X_val_imp   = imputer.transform(X_val)    ← uses training medians
X_test_imp  = imputer.transform(X_test)   ← uses training medians
```

**Why median over mean:**  
Distributions of transit depth, planet radius, and SNR are heavily right-skewed due to eclipsing binary outliers. The mean is pulled toward extreme false positive values. The median sits in the dense region where most transit signals are. Using median avoids imputing unrealistically large values for missing measurements.

**Why fit on training data only:**  
If you compute the median across ALL 3,901 rows and then split, the median used for imputation was computed using information from the validation and test sets. This is data leakage — a subtle form where test set statistics influence preprocessing.

The correct approach: medians are computed from training data only. Val and test sets use the same training medians. This guarantees the test set is completely unseen until final evaluation.

**Why not KNN imputation:**  
KNN imputation finds the k nearest training examples and imputes their average value. More sophisticated — but:
- Much slower for 37 features and 3,901 rows
- Still requires fitting on training data only (same principle)
- For this dataset size and this project scope, median imputation is sufficient

---

## Decision 5 — Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# CRITICAL: Fit on training data ONLY
scaler = StandardScaler()
scaler.fit(X_train_imp)    ← compute mean and std from training set only

# Apply (transform) to all sets
X_train_scaled = scaler.transform(X_train_imp)
X_val_scaled   = scaler.transform(X_val_imp)    ← uses training statistics
X_test_scaled  = scaler.transform(X_test_imp)   ← uses training statistics
```

**What StandardScaler does:**
```
For each feature f:
X_scaled = (X - mean_f) / std_f

Result: each feature has mean ≈ 0, standard deviation ≈ 1
```

**Why scaling is needed:**  
Different features have wildly different scales:
- `koi_period`: 0.5 to 500 (days)
- `koi_steff`: 3,000 to 8,000 (Kelvin)
- `koi_depth`: 50 to 500,000 (ppm)
- `koi_fpflag_co`: 0 or 1 (binary)

For SVM and Logistic Regression: unscaled features cause the model to weight large-range features (depth, SNR) much more than small-range features (binary flags). Scaling puts all features on equal footing.

For the Genesis CNN: unscaled inputs cause activation values to be dominated by high-range features, making gradient flow unstable.

**Why StandardScaler over MinMaxScaler:**  
MinMaxScaler maps each feature to [0, 1] using min and max. But in this dataset, the min and max are often dominated by extreme false positive outliers (max koi_depth = 500,000 ppm). Mapping to [0, 1] with these extremes would compress 99% of the data into the [0, 0.02] range.

StandardScaler is outlier-affected (std is inflated by extremes) but the effect is milder — it compresses the main distribution slightly rather than severely.

**Why RobustScaler was not chosen:**  
RobustScaler uses median and IQR instead of mean and std — genuinely robust to outliers. It would handle the extreme false positive values better. However, StandardScaler is the standard in the exoplanet vetting literature (consistent with papers like Shallue & Vanderburg 2018), and the difference in practice is minor given the non-linear model architectures.

**Why fit on training data only (same principle as imputation):**  
If you compute mean and std across all 3,901 rows and then split, the scaler was fitted using test set statistics. Same leakage risk as imputation. Must fit on training data only.

---

## Decision 6 — Saving Preprocessing Artefacts

```python
import pickle

# Save imputer and scaler for deployment
with open('results/models/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

with open('results/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

**Why save these:**  
When deploying the model to vet new candidates (e.g., from TESS), you need to preprocess new data **exactly the same way** as training data. This means using the same median values (computed from the Kepler training set) and the same scaling statistics.

If you re-fit the imputer and scaler on the new data, you would introduce subtle differences in how the features are scaled, which would degrade model performance.

Saving the trained preprocessing artefacts allows reproducible, consistent preprocessing at deployment time.

---

## The Complete Decision Summary Table

| Decision | Choice | Alternative | Why Chosen |
|----------|--------|------------|------------|
| Missing column threshold | 50% | 30%, 80% | Balance between data loss and invented values |
| Leaky features | Excluded | Included | Prevents trivially easy prediction |
| Imputation strategy | Median | Mean, KNN | Right-skewed distributions; mean pulled by EBs |
| Imputer fitting | Train only | All data | Prevents test set leakage |
| Split ratio | 70/15/15 | 80/10/10 | Standard for this dataset size |
| Split type | Stratified | Random | Preserves class ratio in each split |
| Scaling method | StandardScaler | RobustScaler, MinMaxScaler | Consistent with literature |
| Scaler fitting | Train only | All data | Prevents test set leakage |
| Class weighting | balanced | None, manual | Corrects 4.9× imbalance systematically |
| Label encoding | Integer (0/1/2) | One-hot | Standard for sklearn; one-hot handled in Keras |

---

*Previous: [09 — Feature Engineering Decisions](09_feature_engineering_decisions.md)*  
*Next: [11 — Glossary](11_glossary.md)*
