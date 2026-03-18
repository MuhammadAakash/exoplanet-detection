# 08 — EDA Key Findings and What They Mean for Modelling

> This is the "so what" file. Everything the EDA found, and exactly how each finding changed what I did next.

---

## The 8 Key Findings — At a Glance

| Finding | What I Found | What I Did |
|---------|-------------|-----------|
| 1. Class Imbalance | 4.9× ratio (CONFIRMED to CANDIDATE) | `class_weight='balanced'`, macro metrics, stratification |
| 2. Missing Values | Physically motivated gaps, not random | Median imputation, 50% drop threshold, train-only fitting |
| 3. FP Flags Dominate | Top 3 ANOVA features are FP flags | Expect flags to dominate RF importances; CNN must exploit them |
| 4. Physics Is in the Data | Depth-period plot separates classes visually | Dataset is valid; learning is grounded in real astronomy |
| 5. Correlated Features | 6 high-correlation pairs identified | Dual-branch CNN designed for both local and broad patterns |
| 6. Outliers Are Genuine FPs | 200 R⊕ "planet" is an eclipsing binary | Do NOT remove outliers; they are the most extreme false positives |
| 7. CANDIDATE Overlaps Both | CANDIDATE sits between CONFIRMED and FP in all features | Expect lowest CANDIDATE F1; this is scientifically correct |
| 8. Labels Are Trustworthy | koi_score separates classes cleanly | Training data is reliable; proceed to modelling with confidence |

---

## Finding 1 — Class Imbalance (4.9× ratio)

### What I found
```
CONFIRMED:      2,341 (60.0%)
FALSE POSITIVE: 1,083 (27.8%)
CANDIDATE:        477 (12.2%)

Imbalance ratio: 4.9× (largest to smallest class)
```

### Why a 4.9× imbalance is significant
A model that predicts CONFIRMED for every input achieves 60% accuracy. This is called the **majority class baseline** — a completely dumb model that learns nothing. Without correcting for imbalance, a real model might learn this shortcut rather than the genuine classification patterns.

### What I did because of this

**Balanced class weighting:**
```python
# sklearn models:
RandomForestClassifier(class_weight='balanced')
LogisticRegression(class_weight='balanced')
SVC(class_weight='balanced')

# Keras model:
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
```

The `'balanced'` mode sets each class weight inversely proportional to its frequency:
```
Weight for class c = total_samples / (n_classes × count_of_class_c)

CONFIRMED:      3901 / (3 × 2341) = 0.556
FALSE POSITIVE: 3901 / (3 × 1083) = 1.200
CANDIDATE:      3901 / (3 ×  477) = 2.727
```
CANDIDATE training examples are weighted ~4.9× more than CONFIRMED — exactly countering the frequency imbalance.

**Evaluation metrics used:**
- F1-macro: average F1 across all three classes, treating each equally
- Cohen's Kappa: agreement corrected for chance (given class frequencies)
- Matthews Correlation Coefficient (MCC): single metric capturing all cells of confusion matrix
- NOT: accuracy alone (meaningless given imbalance)

**Stratified splits:**
```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
```

Result:
- Training: 2,730 samples (60/28/12 class ratio preserved)
- Validation: 585 samples (same ratio)
- Test: 586 samples (same ratio)

---

## Finding 2 — Missing Values Are Physically Motivated

### What I found

**Columns dropped (> 50% empty):** 15 columns removed including `koi_eccen`, `koi_longp`, `koi_ingress`, and others based on circular orbit assumption.

**Columns with partial missingness in 37 model features:**
- `koi_smet`: ~30% missing (no spectroscopy for faint stars)
- `koi_bin_oedp_sig`: ~20% missing (too few transits to compute)
- `koi_slogg`, `koi_srad`, `koi_smass`: 5–15% missing

**The critical insight:** Missing values are NOT random. `koi_smet` is missing more often for fainter stars. `koi_bin_oedp_sig` is missing more often for candidates. This is missing-not-at-random (MNAR).

### What I did because of this

**Median imputation (not mean):**

Why median: The distributions of transit depth, planet radius, and SNR are heavily right-skewed due to extreme eclipsing binary values:
```
Example — koi_depth:
Q1   (25th percentile):   ~500 ppm    ← where most planets are
Q2   (50th percentile):  ~1,200 ppm   ← median — good imputation value
Mean:                   ~8,000 ppm    ← pulled toward EB extremes
Q3   (75th percentile):  ~4,000 ppm
Max:                  ~500,000 ppm    ← eclipsing binary extreme
```

Imputing with mean would give ~8,000 ppm — which implies a large Jupiter-sized planet. Most missing transit depths come from faint or short-cadence targets where a typical Earth-to-Neptune-size planet is more likely. Median (1,200 ppm) is a much better placeholder.

**Train-only fitting:**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)           # Learn medians from training data only
X_train = imputer.transform(X_train)
X_val   = imputer.transform(X_val)    # Apply same medians to val/test
X_test  = imputer.transform(X_test)   # No leakage from test set
```

Fitting on all data would let test set median values influence the imputation — a subtle form of leakage that would make test performance slightly optimistic.

---

## Finding 3 — False Positive Flags Are Overwhelmingly Powerful

### What I found
```
Top ANOVA F-scores:
koi_fpflag_co: 2,197  ████████████████████████████████████████████████████████████
koi_fpflag_ss: 1,357  █████████████████████████████████████
koi_fpflag_ec:   695  ██████████████████████
─────────────────────────────────────────────────────────────────
koi_incl:        505  ██████████████  (next best non-flag feature)
koi_teq:         286  ████████
koi_num_transits:279  ████████
```

The gap between the flags and everything else is enormous. The three FP flags are in a fundamentally different discriminative league.

### Why this is not surprising in retrospect
These flags are not statistical correlations — they are **direct physical diagnostics** designed specifically to detect false positives. When `koi_fpflag_ss = 1`, an eclipsing binary is almost definitionally identified. No amount of transit geometry measurement can be more definitive than that.

### What this means for modelling

**For Random Forest:**
The RF feature importances should match the ANOVA ranking. If `koi_fpflag_co` is not the most important RF feature, something unexpected is happening that needs investigation.

**For the Genesis CNN:**
The CNN processes features as a 1D sequence. These four flags must be arranged in the feature vector such that the convolutional kernel can "see" them. Since they are related (all are FP diagnostics), placing them adjacent in the feature vector means both kernel sizes (3 and 7) will capture their interactions.

**For dissertation:**
"The ANOVA analysis predicted that FP flags would dominate classification. The Random Forest confirmed this post-training. This consistency between pre-modelling analysis and post-modelling feature importances validates that the model is learning genuine astrophysical patterns, not statistical artefacts."

---

## Finding 4 — The Physics Is Directly Visible in the Data

### What I found

On the period-depth scatter plot (log-log scale):
- FALSE POSITIVES cluster at depths > 10,000 ppm (stellar-sized eclipses)
- CONFIRMED planets cluster at depths < 10,000 ppm (planetary-sized transits)
- CANDIDATES sit in the overlap region

On the planet radius plot:
- A natural boundary at ~11–15 R⊕ separates confirmed planets from false positives
- FALSE POSITIVES scatter to 50–300 R⊕ (stellar companion radii)
- CONFIRMED stay below 15 R⊕ (physical planet size limit)

### Why this finding matters for the model

It validates that the supervised learning problem is **physically grounded**. The model is not finding arbitrary statistical patterns in numerical columns. It is learning physical distinctions that correspond to real differences in the underlying astrophysics.

This is important for the dissertation because it answers the potential examiner question: "How do you know your model is learning astronomy and not memorising noise?"

The answer: "Because the class boundaries in feature space align with known physical boundaries — the Jupiter radius limit, the stellar eclipse depth range, the secondary eclipse constraint. These are not learned by the model from data alone; they are imposed by the physics of what stars and planets are."

---

## Finding 5 — Correlated Feature Pairs Exist (Physically Expected)

### What I found

Highly correlated pairs in the model features:
```
koi_prad  ↔ koi_ror:    r ≈ 0.95  (prad = ror × srad — mathematical)
koi_depth ↔ koi_ror:    r ≈ 0.90  (depth = ror² — mathematical)
koi_sma   ↔ koi_period: r ≈ 0.85  (Kepler's Third Law)
koi_teq   ↔ koi_sma:    r ≈ -0.85 (Teq ∝ a^(-0.5))
koi_insol ↔ koi_teq:    r ≈ 0.90  (both measure stellar irradiation)
koi_steff ↔ koi_srad:   r ≈ 0.65  (HR diagram: hotter = larger)
```

### What the correlation patterns mean for architecture

The correlated features cluster in the feature vector when features are ordered by group:
- Transit features: `koi_depth`, `koi_ror`, `koi_prad` adjacent (all measure size)
- Orbital features: `koi_period`, `koi_sma`, `koi_teq`, `koi_insol` adjacent (all measure orbital properties)

The Genesis CNN dual-branch design captures this:
- **Branch 1 (kernel=3):** Captures local interactions — adjacent correlated features (depth, ror, prad)
- **Branch 2 (kernel=7):** Captures broader patterns — longer-range feature relationships (period → sma → teq → insol chain)

This is why removing correlated features would hurt the CNN — it was designed to integrate these correlated groups.

---

## Finding 6 — Outliers Are Genuine False Positives

### What I found
```
Features with highest outlier rates (> 3 standard deviations):
koi_prad:       ~15% of values beyond 3σ
koi_depth:      ~12% of values beyond 3σ
koi_ror:        ~10% of values beyond 3σ
koi_max_mult_ev: ~8% of values beyond 3σ
```

Most extreme values belong to FALSE POSITIVE class.

### The insight that changes everything
A `koi_prad` of 200 Earth radii is not a measurement error. It is the correctly computed "planet radius" for a companion star that the Kepler pipeline incorrectly modelled as a planet. This value is physically real and scientifically informative — it is one of the clearest indicators in the entire dataset that this KOI is a false positive.

Removing it would remove a highly informative training example.

### What I did
Kept all outliers. Used StandardScaler (not RobustScaler) for consistency with literature.

Acknowledged the limitation: StandardScaler's standard deviation is inflated by outliers, slightly compressing the main distribution. But the model's non-linear activation functions handle this compression adequately.

---

## Finding 7 — CANDIDATE Class Genuinely Overlaps Both Others

### What I found
In every feature distribution plot, the CANDIDATE class sits between CONFIRMED and FALSE POSITIVE — sometimes overlapping significantly with both. There is no feature (not even the FP flags) that cleanly separates CANDIDATE from both other classes simultaneously.

From the FP flag perspective:
- Confirmed: all four flags = 0
- False positive: one or more flags = 1
- Candidate: all four flags = 0 (same as confirmed!)

The FP flags cannot distinguish CANDIDATE from CONFIRMED — they both have clean flag profiles. The difference between these two classes must be found in the continuous features (transit depth, number of transits, stellar parameters) and their combinations.

### What this means for model expectations

CANDIDATE will have the lowest per-class F1 score in every model. This is correct and expected — not a model failure.

```
Expected per-class F1 ordering (all models):
FALSE POSITIVE > CONFIRMED > CANDIDATE

Reason:
False positive: Strong flag features make it easy to detect
Confirmed:      Many training examples, distinct patterns
Candidate:      Fewest examples, overlaps with confirmed,
                genuinely uncertain by definition
```

**Do not try to "fix" low CANDIDATE F1 by over-fitting to the CANDIDATE class.** The genuine scientific uncertainty of these objects means perfect classification is not achievable or even correct.

---

## Finding 8 — Labels Are Validated by Independent Robovetter Score

### What I found
```
koi_score distribution by class:
CONFIRMED:      tightly clustered near 1.0 (mean ~0.85, little spread)
FALSE POSITIVE: tightly clustered near 0.0 (mean ~0.10, little spread)
CANDIDATE:      spread across 0.4–0.9 (reflecting genuine uncertainty)
```

The independent Robovetter agrees with the human labels in the vast majority of cases.

### What this proves

**Labels are reliable:** The human-assigned dispositions are internally consistent with independent automated analysis. Training on these labels is valid.

**Features carry genuine information:** The Robovetter uses the same 37-feature-equivalent data to compute its score. The fact that the score separates classes cleanly proves the features have discriminative power — they are not noise.

**CANDIDATE uncertainty is genuine:** Candidates have spread-out koi_score values, not because they are mislabelled but because they are scientifically uncertain.

**Proceed with confidence:** The dataset quality is sufficient for reliable supervised learning.

---

## The Complete Decision Chain

```
EDA Finding                     → Modelling Decision

Class imbalance (4.9×)          → class_weight='balanced' in all models
                                   stratified splits
                                   macro F1, kappa, MCC as metrics

Missing values (physically       → median imputation (not mean)
motivated, MNAR)                   50% threshold for column dropping
                                   fit imputer on training data only

FP flags dominate (F > 700)     → expect flags = top RF importances
                                   feature arrangement in CNN input
                                   dual-branch CNN for flag interactions

Physics visible in data         → model is learning real astronomy
                                   dissertation validation argument

Correlated features             → dual-branch CNN (kernel 3 + kernel 7)
(physically expected)              do NOT remove correlated features

Outliers = genuine FPs          → do NOT remove outliers
                                   use StandardScaler (not removal)
                                   acknowledge outlier inflation

CANDIDATE overlaps both         → expect lowest CANDIDATE F1
                                   do not over-fit to minority class
                                   macro metrics capture this correctly

Labels validated by             → trust training data
koi_score                          dataset quality is not a confounder
```

---

*Previous: [07 — EDA What I Explored and Why](07_eda_what_i_explored_and_why.md)*  
*Next: [09 — Feature Engineering Decisions](09_feature_engineering_decisions.md)*
