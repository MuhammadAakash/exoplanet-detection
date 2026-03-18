# 09 — Feature Engineering Decisions

> Why I added 3 new features, why I kept everything else as-is, and why I deliberately did NOT do several things that might seem reasonable.

---

## What Feature Engineering Is

Feature engineering is creating new features from existing ones to give your model better information than the raw measurements provide.

The classic example: you have a person's **weight** and **height** separately. BMI = weight / height² combines them into a single feature that is far more medically informative than either measurement alone. Feature engineering is creating that BMI equivalent.

---

## Why This Dataset Needs Very Little Engineering

The most important thing to understand about this dataset: **NASA's pipeline already did the feature engineering.**

When Kepler detected a transit signal, NASA did not give me raw pixel brightness values. It gave me **already-computed physical measurements** — orbital period, transit depth, planet radius, stellar density, signal-to-noise ratio. Each of these required sophisticated astrophysical model fitting applied to the raw light curves.

In other words, the 37 features I am using are the **engineered outputs** of NASA's own pipeline, not the raw inputs. The pipeline already extracted the physically meaningful quantities from the raw data.

This is fundamentally different from working with, say, raw text (where you need to extract word frequencies) or raw images (where you need to extract edges and textures). The domain experts (NASA) have already done the feature extraction work.

So my feature engineering is **targeted and minimal** — three specific additions with clear physical motivation.

---

## The Three Engineered Features

### Feature 1 — Combined False Positive Flag Score

```python
df['fp_flag_sum'] = (
    df['koi_fpflag_nt'] +
    df['koi_fpflag_ss'] +
    df['koi_fpflag_co'] +
    df['koi_fpflag_ec']
)
# Range: 0 (no flags — planet-like) to 4 (all flags — almost certainly not a planet)
```

#### Why I created this

The four individual flags tell you **which** automated tests failed. The sum tells you **how many** tests failed simultaneously.

Think of it this way: if a doctor orders four blood tests and one comes back abnormal, it is concerning. If all four come back abnormal, it is a medical emergency — not just "four times as concerning" but qualitatively different.

The same logic applies here. A KOI with 3 flags set is not merely 3× more suspicious than one with 1 flag — multiple independent tests pointing toward false positive is **qualitatively stronger evidence**. The model has to learn this cumulative effect implicitly from four separate binary columns. Giving it explicitly as a single feature makes learning faster and more reliable.

Physical meaning of each value:
```
0: Passed all four tests — planet-like signal
1: Failed one test — suspicious, follow-up warranted
2: Failed two independent tests — strong evidence of false positive
3: Failed three tests — very strong evidence, almost certainly not a planet
4: Failed all four tests — definitively false positive
```

#### Why this helps especially for classical ML

Logistic Regression is a linear model. For it to learn the cumulative effect of flags, it would need to find the right combination of four separate binary coefficients. Giving it the sum means it can learn the cumulative effect with a single coefficient. Simple, efficient, and correct.

---

### Feature 2 — Physically Impossible Size Indicator

```python
df['size_flag'] = (df['koi_prad'] > 15).astype(int)
# 0: physically possible planet (≤ 15 Earth radii)
# 1: physically impossible planet size (> 15 Earth radii)
```

#### The physics behind this

Nothing larger than approximately 11–15 Earth radii can be a planet. At this size boundary (1 Jupiter radius), objects transition from planets to brown dwarfs to stellar companions. The boundary is physically hard — it is set by the minimum mass required for deuterium fusion (brown dwarf threshold) and ultimately by the hydrogen-burning minimum mass for stars.

```
Size boundary reference:
< 1.5  R⊕: Rocky super-Earth (almost certainly planet)
1.5–4  R⊕: Mini-Neptune / Neptune
4–11.2 R⊕: Sub-Jupiter / Jupiter
> 11.2 R⊕: Brown dwarf or star (NOT a planet)
─────────────────────────────────────────────────────
Threshold chosen: 15 R⊕ (conservative, allows for measurement uncertainty)
```

#### Why this helps when `koi_prad` already exists

`koi_prad` is a continuous variable. The model has to discover that values above ~15 are qualitatively different from values below — a non-linear threshold. Random Forest discovers this automatically (it can make threshold splits). Logistic Regression CANNOT learn this non-linear boundary on its own without polynomial feature engineering.

By creating an explicit binary flag at the physical boundary, I encode domain knowledge directly into the feature space — making it accessible to all models including linear ones.

#### Why 15 rather than 11.2

The true boundary (1 Jupiter radius = 11.2 R⊕) is the theoretical minimum. But `koi_prad` has measurement uncertainty — a 10 R⊕ planet could have true radius 12 R⊕ given the stellar radius uncertainty. Using 15 gives a conservative buffer that captures definitively impossible sizes while avoiding flagging genuine massive planets due to measurement error.

---

### Feature 3 — Per-Transit Signal Strength

```python
df['snr_per_transit'] = df['koi_model_snr'] / (df['koi_num_transits'] + 1)
# +1 prevents division by zero for KOIs with 0 recorded transits
```

#### The problem with total SNR

Total SNR (`koi_model_snr`) accumulates with each additional transit:
```
SNR_total ≈ SNR_per_transit × √N_transits
```

A planet seen 100 times naturally has SNR ≈ 10× a planet seen 1 time — even if each individual transit is equally clear. This means `koi_model_snr` conflates two different things: **how clearly each transit was seen** and **how many transits accumulated**.

For a confirmed planet with many transits, high total SNR is expected. For a candidate with few transits, even moderate total SNR means each individual transit was very clearly detected.

#### What per-transit SNR reveals

A genuine planet should have consistent depth (and therefore consistent per-transit SNR) across all transits. Consistency is evidence of a periodic, stable signal.

A suspicious signal might have high total SNR because ONE particularly strong event dominated — a cosmic ray hit, a stellar flare, a satellite crossing. In this case:
```
koi_max_sngle_ev >> koi_model_snr / √N_transits

Per-transit SNR = total SNR / N_transits would be high
But the single event statistic (koi_max_sngle_ev) would also be suspiciously high
```

The per-transit SNR, combined with `koi_max_sngle_ev`, helps the model detect these single-event false positives.

#### The +1 denominator

Without the +1, KOIs with `koi_num_transits = 0` would produce division by zero. The +1 is a standard smoothing technique that avoids this edge case. It has negligible effect for KOIs with many transits (100 / 101 ≈ 99/100) but prevents crashes for the rare edge case.

---

## What I Deliberately Did NOT Do

### Did NOT create polynomial features

Tempting approach: create `period²`, `depth × duration`, `period × ror`, etc. to capture non-linear combinations.

**Why not:**
- Random Forest and the Genesis CNN already learn non-linear combinations implicitly
- For Random Forest: each split is a threshold on a single feature; combinations of splits across multiple trees capture any interaction
- For CNN: convolutional filters over the feature sequence learn combinations of adjacent features
- Creating dozens of polynomial features would risk overfitting on a dataset of 3,901 samples
- Would make the model harder to interpret and explain in the dissertation

### Did NOT apply log transformation as separate features

Tempting approach: add `log(koi_period)`, `log(koi_depth)`, `log(koi_prad)` as additional features alongside the originals.

**Why not:**
- For Random Forest: threshold splits on a feature are equivalent to threshold splits on any monotonic transformation of that feature. `log(depth) > 4` is exactly equivalent to `depth > 54.6`. The Forest does not benefit from both forms.
- For CNN: StandardScaler already compresses the range of skewed features, making log transformation less critical
- Adding both original and log versions effectively halves the model's feature budget for new information

### Did NOT remove correlated features

The EDA showed high correlations: `koi_depth` ↔ `koi_ror` (r=0.90), `koi_prad` ↔ `koi_ror` (r=0.95), `koi_period` ↔ `koi_sma` (r=0.85).

Tempting approach: for each correlated pair, remove one feature to reduce redundancy.

**Why not:**
- Correlated features still carry slightly different information (ror is model-independent; prad requires stellar radius knowledge — different measurement error characteristics)
- Random Forest handles correlated features gracefully by splitting importances between them
- The Genesis CNN dual-branch architecture was explicitly designed to integrate correlated features through branches of different kernel sizes. Removing features would undermine this design.
- These correlations are physically expected (Kepler's Third Law, transit geometry mathematics) — they validate the data rather than indicating a problem

### Did NOT do PCA dimensionality reduction

Tempting approach: reduce 37 correlated features to a smaller number of uncorrelated principal components.

**Why not:**
- PCA components are linear combinations of all features — they become physically uninterpretable. "Principal Component 1 is 0.23×period + 0.18×depth + 0.31×ror + ..." has no physical meaning in astronomy.
- The dissertation requires connecting model findings to physical feature interpretation (feature importances, ANOVA vs RF importance comparison). PCA would destroy this.
- The correlation structure in my data is physically motivated and informationally meaningful — reducing it would discard genuine physical information.

---

## The Complete Engineered Feature Set

After adding 3 features to the original 37:

```
Total model features: 40

Original 37 features:
  Transit geometry (13): koi_period, koi_time0bk, koi_impact, koi_duration,
                         koi_depth, koi_ror, koi_srho, koi_prad, koi_sma,
                         koi_incl, koi_teq, koi_insol, koi_dor
  Stellar parameters (8): koi_steff, koi_slogg, koi_smet, koi_srad, koi_smass,
                           ra, dec, koi_kepmag
  FP flags (4): koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec
  Signal quality (5): koi_max_sngle_ev, koi_max_mult_ev, koi_model_snr,
                      koi_num_transits, koi_bin_oedp_sig
  Magnitudes (7): koi_gmag, koi_rmag, koi_imag, koi_zmag, koi_jmag,
                  koi_hmag, koi_kmag

Engineered (3):
  fp_flag_sum:     Cumulative FP flag evidence (0–4)
  size_flag:       Physically impossible planet size binary (0/1)
  snr_per_transit: Signal strength per individual transit
```

---

## Where Feature Engineering Sits in the Pipeline

Feature engineering must happen **after** loading raw data but **before** train/val/test split (for features derived from existing columns without using any statistics).

For these three specific features:
- `fp_flag_sum`: Sum of existing binary columns. No statistics needed. No leakage risk. Can be computed before or after splitting.
- `size_flag`: Threshold comparison with a fixed physical constant (15 R⊕). No statistics needed. No leakage risk.
- `snr_per_transit`: Division of two existing columns. No statistics needed. No leakage risk.

All three are computed from existing columns using fixed physical constants or simple arithmetic — no data-derived statistics are needed. There is no leakage risk regardless of when they are computed relative to the split.

Compare this to imputation (compute median from training data → risk of leakage if computed on full dataset) or scaling (compute mean/std from training data → risk of leakage if computed on full dataset). Those operations MUST be fitted on training data only.

---

## How to Write This in the Dissertation

> "Feature engineering was deliberately minimal for this dataset. The NASA KOI catalogue already contains extensively engineered astrophysical features — the result of sophisticated model fitting applied to raw Kepler light curves — making additional engineering largely unnecessary. Three physically motivated features were added: a cumulative false positive flag score (0–4) encoding the total number of failed automated vetting tests, a binary indicator for physically impossible planet radii above 15 Earth radii, and a per-transit SNR metric normalising signal strength by the number of observed transits. These additions encode specific domain knowledge that would otherwise require non-linear interactions to learn implicitly. No polynomial features, log transformations, or dimensionality reduction were applied, as these would either provide no additional information to ensemble and deep learning models or destroy the physical interpretability essential for dissertation analysis."

---

*Previous: [08 — EDA Key Findings and Decisions](08_eda_key_findings_and_decisions.md)*  
*Next: [10 — Preprocessing Decisions](10_preprocessing_decisions.md)*
