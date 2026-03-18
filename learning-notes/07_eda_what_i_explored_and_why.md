# 07 — EDA: What I Explored and Why (All 8 Sections)

> EDA is not a formality before modelling. It is the process of understanding your data before asking a machine to learn from it. Every section answers a specific question that shapes a downstream decision.

---

## The Philosophy of This EDA

Before writing a single line of model code, I needed to answer one master question:

> **"What is my data telling me, and will my model be able to learn from it?"**

The eight EDA sections each answer a specific sub-question:

```
Section 1 → "Are my classes balanced?"
Section 2 → "What data is missing and why?"
Section 3 → "Does each feature look different across classes?"
Section 4 → "Are any features just saying the same thing as each other?"
Section 5 → "Does the physics of astronomy actually show up in the data?"
Section 6 → "How extreme are the extreme values, and should I remove them?"
Section 7 → "Which features are most promising before I train anything?"
Section 8 → "Can I trust my labels?"
```

**Every finding connects to a modelling decision.** The EDA is not exploratory for its own sake — it is evidence-gathering that directly justifies the choices made in preprocessing, model design, and evaluation.

---

## Section 1 — Class Distribution

### The question
"Are my three classes balanced, or does one dominate?"

### Why this is the first thing to check
You cannot make a single modelling decision without knowing the class balance. The entire evaluation strategy changes depending on the answer.

### What I did
Counted the instances of each class. Created a bar chart (absolute counts) and a pie chart (proportions) in a single figure.

Two panels because they tell different stories:
- Bar chart: shows raw numbers → how many examples does each class have for training?
- Pie chart: shows proportions → how badly is the dataset skewed?

### What I found
```
CONFIRMED:      2,341  (60.0%)  ████████████████████████████████████████
FALSE POSITIVE: 1,083  (27.8%)  ████████████████████
CANDIDATE:        477  (12.2%)  █████████

Imbalance ratio: 4.9× (most common to least common class)
```

### What this drove (the modelling decisions)

**Decision 1 — Class weights:**  
All sklearn models use `class_weight='balanced'`. This makes the model pay equal attention to all three classes regardless of their frequency. Without this, the model would effectively ignore CANDIDATE (the smallest class) and focus on correctly classifying CONFIRMED (the largest).

**Decision 2 — Metrics:**  
Accuracy is useless here. A model that predicts CONFIRMED for everything achieves 60% accuracy without learning a single pattern. Primary metrics: F1-macro (treats all classes equally), Cohen's Kappa (corrects for chance agreement given imbalance), MCC (most informative single metric for imbalanced multiclass).

**Decision 3 — Stratified splits:**  
Every train/val/test split preserves the 60/28/12 ratio. Without stratification, random chance might put all candidates in train and leave none for test, or vice versa.

**Decision 4 — Expectation setting:**  
CANDIDATE will have the lowest per-class F1 across all models — not because the models are failing, but because it is the hardest class (fewest examples, most genuine ambiguity). This is expected and correct.

---

## Section 2 — Missing Value Analysis

### The question
"What data is absent, and can I trust what remains?"

### Why this matters more than it sounds
In most machine learning tutorials, missing values are treated as a nuisance to be dealt with quickly. In this dataset, the missingness is **physically motivated** — it tells you something about how the data was collected and what the missing values would have contained.

Understanding WHY data is missing determines HOW to handle it.

### What I did
Two sub-plots:

**Plot A — All columns in raw dataset:**  
Bar chart sorted by missing percentage. Coloured green (< 50%, keep and impute) vs red (> 50%, drop). The 50% threshold is where you cross from "more measured than invented" to "more invented than measured."

**Plot B — The 37 model features by group:**  
Heatmap of missing percentages for the features I actually use, grouped by feature type. Shows which groups of features have systematic missingness patterns.

### What I found — three types of missingness

**Type 1 — Missing by physical assumption (drop these columns):**  
`koi_eccen`, `koi_longp`, `koi_ingress` — nearly 100% empty because NASA assumed circular orbits. These columns were never filled in for most KOIs. Keeping them would add columns that are almost entirely invented values.

**Type 2 — Missing due to observational limits (impute with median):**  
`koi_smet` (~30% missing), `koi_slogg` (~10% missing), `koi_srad` (~10% missing) — missing because faint stars never received spectroscopic follow-up. This is **missing-not-at-random** — the missing cases are systematically different (fainter, less studied) from the measured cases. Median imputation is imperfect but necessary.

**Type 3 — Missing due to insufficient data (impute with median):**  
`koi_bin_oedp_sig` — missing for KOIs with < 4 transits, because the odd-even test requires at least 2 transits of each parity. Missing here means "not computable" — which itself is informative about the number of transits.

### The imputation decisions and their justifications

**Why median over mean:**  
Many features are right-skewed due to extreme false positive outliers. Transit depth ranges from 84 ppm (Earth-like) to 500,000 ppm (eclipsing binary). The mean gets pulled toward those extremes. The median sits in the dense central region where most genuine signals are. Imputing with median gives a more physically sensible placeholder.

**Why fit imputer on training data only:**  
If you compute the median across all 3,901 rows (including test), you have used information from the test set in your preprocessing. The test set becomes slightly contaminated — a subtle but real form of data leakage. The correct procedure: fit the imputer on training data only, then apply (transform) to val and test separately.

**Why not drop rows with missing values:**  
With missing values spread across multiple features, dropping any row with any missing value would lose a large fraction of training data. From 2,730 training examples, you might fall below 1,500 — insufficient for reliable model training.

---

## Section 3 — Feature Distributions by Class

### The question
"Does each feature actually look different for confirmed planets versus false positives versus candidates?"

### Why this is the core of EDA for classification
A feature is only useful for classification if it looks different across classes. This section is "looking at each feature through a microscope" before handing it to a model.

If a feature's distributions for all three classes overlap perfectly → useless for classification  
If a feature's distributions are in different regions → useful for classification

### What I did — three different plot types for three feature groups

**Transit features → KDE plots (13 features):**  
Kernel Density Estimation produces smooth continuous curves showing the full distribution shape including tails and modes. Better than histograms for capturing the full shape of these wide-ranging, skewed distributions.

Plotted at 99th percentile cutoff to prevent extreme outliers from collapsing the scale. The 1% of values above the cutoff still exist — just not plotted, to keep the main distribution visible.

**Stellar features → Box plots (8 features):**  
Stellar parameters like temperature and metallicity are more symmetric and better summarised by median and quartile ranges. Box plots make class comparison cleaner when the distributions are less skewed.

**Signal quality features → KDE plots (5 features):**  
Detection statistics are skewed and span wide ranges — KDE reveals the full shape, which a box plot would compress.

### What I found — selected highlights

**koi_depth (transit depth):**  
Clear class separation. False positives cluster at high depths (10,000+ ppm) due to stellar-sized eclipses. Confirmed planets cluster at low-to-moderate depths (50–5,000 ppm). Candidates span the middle. This is the expected physical pattern.

**koi_num_transits:**  
Confirmed planets have many transits (short-period → many orbits in 4 years). Candidates have few transits (long-period → few orbits). False positives are distributed across all ranges.

**koi_steff:**  
Mild separation. Confirmed planets slightly more common around F-G stars (Kepler's priority targets). False positives spread across all stellar types.

**koi_smet:**  
Mild enrichment in confirmed planet hosts (planet-metallicity correlation), but weak discrimination.

### The statistical summary table
Saved as CSV: mean ± standard deviation for every feature in every class. Provides precise numbers for dissertation text without needing to re-read plots.

---

## Section 4 — Correlation Structure

### The question
"Are any of my 37 features just saying the same thing as another feature?"

### Why correlations matter for this model specifically

**For interpretation:** When two correlated features both appear as "important" in a Random Forest, you need to understand whether they are carrying independent information or splitting the same signal between them.

**For architecture:** The Genesis CNN's dual-branch design (kernel size 3 and 7) was chosen specifically because of the correlation structure. Branch 1 captures short-range local interactions (adjacent correlated features). Branch 2 captures longer-range global patterns.

### What I did
Pearson correlation matrix for all 37 features. Plotted as a heatmap with:
- Hierarchical clustering to group correlated features visually
- Lower triangle only (matrix is symmetric, upper half is redundant)
- Blue-red diverging colormap (blue = positive, red = negative correlation)

### What I found — physically expected correlations

| Feature Pair | Correlation | Physical Law Behind It |
|---|---|---|
| `koi_prad` ↔ `koi_ror` | r ≈ 0.95 | `prad = ror × srad` — mathematical |
| `koi_depth` ↔ `koi_ror` | r ≈ 0.90 | `depth = ror²` — mathematical |
| `koi_sma` ↔ `koi_period` | r ≈ 0.85 | Kepler's Third Law: `a³ ∝ T²` |
| `koi_teq` ↔ `koi_sma` | r ≈ -0.85 | `Teq ∝ a^(-0.5)` — hotter closer in |
| `koi_steff` ↔ `koi_srad` | r ≈ 0.65 | HR diagram: hotter stars are larger |
| `koi_insol` ↔ `koi_teq` | r ≈ 0.90 | Both measure stellar irradiation |

### Why these correlations VALIDATE the dataset
These correlations should be there — they are predicted by physical laws (Kepler's Third Law, Stefan-Boltzmann Law, etc.). Their presence is evidence that the data is physically sensible and the derived quantities are correctly computed.

If `koi_sma` and `koi_period` were uncorrelated, something would be fundamentally wrong with the data.

### Why correlated features are NOT removed
- They carry slightly different information (ror is model-independent; prad requires stellar radius knowledge)
- Random Forest handles correlated features gracefully
- The Genesis CNN architecture was explicitly designed to integrate correlated features through its dual branches

---

## Section 5 — Astrophysical Relationships

### The question
"Does the known physics of exoplanet detection actually show up in my data?"

### Why this section exists — unique to this project

Most ML EDA sections are purely statistical. This section is different: because my data has known physics behind it, I can check whether the patterns match what astronomy predicts. This is both a **sanity check** and a **validation** of the dataset.

### Plot 1 — Period vs Transit Depth (log-log scatter)

**Why log-log:** Both variables span multiple orders of magnitude (period: 0.5–500 days; depth: 50–500,000 ppm). Linear scale would compress 95% of the data into one corner.

**Expected pattern:** False positives should cluster at high depths (stellar-sized dips). Confirmed planets at moderate depths. Candidates spread across middle.

**What I saw:** Exactly this pattern. The three classes separate visually on the log-log plot — without any model. This validates both the dataset and the physical interpretation of transit depth as a size indicator.

### Plot 2 — Planet Radius vs Stellar Radius

**Expected pattern:** Confirmed planets below the ~11.2 R⊕ (Jupiter radius) boundary. False positives scattered above it.

**What I saw:** Clear separation at the physical boundary. False positives with "planet radii" of 50–300 Earth radii are straightforwardly eclipsing binary companions, misinterpreted by the pipeline as planets.

This boundary is not just statistical — it is a **hard physical limit**. No planet can be larger than a Jupiter. Anything larger is not a planet. The model should learn this boundary.

### Plot 3 — Transit Duration vs SNR

**Expected pattern:** Longer transits give more photons and higher SNR (all else equal). But very long transits for short periods are geometrically suspicious (implies very large stars — evolved giant false positives).

### Plot 4 — FP Flag Co-occurrence Heatmap

**Expected pattern:** False positives should have multiple flags co-occurring simultaneously (an eclipsing binary triggers both the secondary eclipse flag AND the not-transit-like flag simultaneously). Confirmed planets should have all flags at zero.

**What I saw:** For false positives: significant co-occurrence of `koi_fpflag_ss` and `koi_fpflag_nt`. For confirmed planets: near-zero co-occurrence of all pairs.

This is physically consistent: an eclipsing binary that has a secondary eclipse (flag_ss = 1) is also likely to have an unusual shape (flag_nt = 1) because the two phenomena have the same underlying cause (it is two stars, not a planet).

---

## Section 6 — Outlier Analysis

### The question
"How extreme are the extreme values, and should I remove them?"

### The key insight — outliers ARE the false positives
In most ML projects: outliers are errors, noise, or irrelevant extremes → remove them.  
In this dataset: outliers are genuine false positive signals → keep them.

A planet radius of 200 Earth radii is a 20-sigma outlier from the planet population. But it is not an error — it is the correctly computed "planet radius" for an eclipsing binary where a stellar companion's radius was misinterpreted as a planet radius. Removing this value removes a genuine false positive training example.

### What I did
For each feature:
1. Computed IQR (Q1 to Q3) — robust spread measure unaffected by outliers
2. Identified values beyond 3-sigma (mean ± 3 standard deviations)
3. Computed the percentage of rows that are outliers by this definition
4. Computed skewness to quantify asymmetry

Plotted outlier percentage for each feature as a horizontal bar chart, sorted by frequency.

### What I found
Features with highest outlier percentages:
- `koi_prad`: ~15% outliers (large "planet radii" from eclipsing binaries)
- `koi_depth`: ~12% outliers (deep dips from stellar eclipses)
- `koi_ror`: ~10% outliers (large radius ratios from stellar companions)
- Signal features: several percent outliers (extreme MES from bright EBs)

Stellar parameters: low outlier rates (stars do not vary as wildly in properties as signals do)

### Why StandardScaler despite its sensitivity to outliers
StandardScaler uses mean and standard deviation. Outliers inflate the standard deviation, slightly compressing the main distribution. This is a known limitation.

Alternatives:
- **RobustScaler** (uses median and IQR): Would handle outliers better
- **StandardScaler** (what I use): Consistent with Kepler vetting literature; outlier compression is mild given the model's non-linear activations

The decision: use StandardScaler, acknowledge the limitation, keep all outliers. The model's non-linear activation functions can handle moderately compressed inputs.

---

## Section 7 — Class Separability (ANOVA F-Scores)

### The question
"Before training any model, which features are most promising, and does this match the physics?"

### Why do this before modelling?

Three reasons:

**Reason 1 — Validation of domain knowledge:**  
Physics predicts FP flags should be most powerful. ANOVA either confirms this (reassuring) or contradicts it (alarming — something is wrong).

**Reason 2 — Prediction of model behaviour:**  
Top ANOVA features should become top Random Forest importances. If they do: consistent. If they differ: the RF found non-linear interactions ANOVA missed.

**Reason 3 — Dissertation narrative:**  
"Before any model was trained, EDA predicted these features would dominate. After training, the model confirmed this. This consistency validates that the model is learning genuine astrophysical patterns."

### What ANOVA F-Score measures
Ratio of between-class variance to within-class variance:
```
F = (variance EXPLAINED by class membership) / (variance WITHIN classes)

High F: feature looks very different across classes → strong discriminator
Low F: feature looks similar regardless of class → weak discriminator
```

### What I found — top 10 features by ANOVA F-score

| Rank | Feature | F-Score | Physical Reason |
|------|---------|---------|----------------|
| 1 | `koi_fpflag_co` | 2,197 | Centroid offset definitionally identifies BEBs |
| 2 | `koi_fpflag_ss` | 1,357 | Secondary eclipse definitionally identifies EBs |
| 3 | `koi_fpflag_ec` | 695 | Known contamination source — reliable match |
| 4 | `koi_incl` | 505 | Proxy for transit shape (EBs cause unusual fits) |
| 5 | `koi_teq` | 286 | Observational bias: hot planets easier to confirm |
| 6 | `koi_num_transits` | 279 | More transits = more power to confirm or rule out |
| 7 | `koi_smet` | 161 | Planet-metallicity correlation |
| 8 | `koi_steff` | 91 | Different stellar populations have different FP rates |
| 9 | `koi_max_mult_ev` | 75 | Detection significance |
| 10 | `koi_max_sngle_ev` | 73 | Single vs multi-event consistency |

Gap between rank 3 (F=695) and rank 4 (F=505) is notable but the dominant gap is between the three FP flags (F≈700–2,200) and everything else. The flags are in a completely different league.

### The violin plots
Top 6 features shown as violin plots by class. Violin plots are chosen over box plots because:
- Box plots show median and quartiles (5 numbers)
- Violin plots show the FULL distribution shape including bimodality and long tails

For binary flags: the violin is almost entirely at 0 for CONFIRMED and at 1 for FALSE POSITIVE — the most visually clean separation possible.

---

## Section 8 — KOI Score Data Quality Validation

### The question
"Can I trust the labels I am training on?"

### Why this is the most overlooked but critical check

Every supervised learning model is only as good as the quality of its labels. If the CONFIRMED labels are unreliable — if some genuine false positives were accidentally labelled as CONFIRMED — the model learns the wrong patterns. And you cannot detect this from model performance alone if the test labels are equally unreliable.

Before trusting the dataset enough to train on it, you need independent evidence that the labels are correct.

### What `koi_score` is
The Robovetter's automated confidence score:
- 1.0 = automated system is certain this is a planet
- 0.0 = automated system is certain this is a false positive
- 0.4–0.9 = automated system is uncertain

Computed by NASA's Robovetter pipeline **completely independently** from the human-assigned `koi_disposition` labels.

### Why `koi_score` is excluded from model features
Label leakage. The Robovetter score was computed by the same NASA pipeline that generated the disposition labels. Including it would mean the model copies the Robovetter's answer rather than independently learning astrophysical patterns from the 37 physical features. The research question would become meaningless.

But analysing it in the EDA costs nothing and buys confidence in the labels.

### What I expected to see (and did see)
```
CONFIRMED:      koi_score distribution tightly clustered near 1.0
FALSE POSITIVE: koi_score distribution tightly clustered near 0.0
CANDIDATE:      koi_score distribution spread across 0.4–0.9
```

### What this proves

1. **Human labels match automated analysis:** The independent Robovetter agrees with the human disposition in the vast majority of cases. This means the labels are internally consistent.

2. **Features carry genuine information:** The Robovetter computes its score from the same physical features I am using. The fact that the score separates classes cleanly means the features ARE informative — if they were noise, the Robovetter could not compute a meaningful score from them.

3. **CANDIDATE class is genuinely uncertain:** Candidates have spread-out koi_score values (not clustered at 0 or 1), confirming they represent real uncertainty — not mislabelled data or lazy labelling.

4. **The supervised learning problem is well-posed:** The training data is trustworthy. I can proceed to modelling with confidence.

---

## How All 8 Sections Connect

```
Section 1 (Class distribution)
    → Sets evaluation strategy: class_weight, macro metrics, stratification
    
Section 2 (Missing values)
    → Sets preprocessing strategy: which columns to drop, median vs mean, fit on train only

Section 3 (Distributions by class)
    → Confirms features carry discriminating information
    → Reveals which features show strongest class separation visually

Section 4 (Correlations)
    → Informs CNN architecture: dual branch handles both local and global correlations
    → Provides physically expected structure to validate dataset

Section 5 (Astrophysical relationships)
    → Validates dataset physically: patterns match known astronomy
    → Provides interpretive framework for model results

Section 6 (Outliers)
    → Justifies keeping outliers (they are genuine false positives)
    → Justifies StandardScaler choice despite outlier sensitivity

Section 7 (ANOVA separability)
    → Predicts model feature importances
    → Creates pre/post modelling narrative for dissertation

Section 8 (KOI score validation)
    → Validates label trustworthiness
    → Rules out dataset quality as an explanation for model failures
```

---

*Previous: [06 — Signal Quality and Magnitude Features](06_feature_physics_signal_and_magnitudes.md)*  
*Next: [08 — EDA Key Findings and Modelling Decisions](08_eda_key_findings_and_decisions.md)*
