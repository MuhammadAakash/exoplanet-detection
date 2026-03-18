# 06 — Feature Physics: Signal Quality & Photometric Magnitude Features

> Signal quality features describe HOW CLEARLY the transit was detected. Magnitude features describe HOW BRIGHT the star is across different wavelengths. Neither group directly tells you WHAT the signal is — but both provide crucial context.

---

## Part 1 — Signal Quality Features (5 features)

| Feature | What It Measures | Unit |
|---------|-----------------|------|
| `koi_max_sngle_ev` | Strongest single individual transit | sigma (σ) |
| `koi_max_mult_ev` | Combined significance of all transits | sigma (σ) |
| `koi_model_snr` | Transit model signal-to-noise ratio | dimensionless |
| `koi_num_transits` | How many transits were observed | count |
| `koi_bin_oedp_sig` | Alternating depth test significance | sigma (σ) |

---

### `koi_max_mult_ev` — Maximum Multiple Event Statistic (MES)

**ANOVA F-Score: 74.5**

#### What it is
The core detection statistic of the Kepler pipeline. When you stack all observed transits at the measured period, MES is the combined signal-to-noise:

```
MES = (transit depth × √N_transits) / noise_per_transit

Where:
N_transits = number of observed transits
noise_per_transit = photometric noise per transit duration
```

Think of MES as a z-score — how many standard deviations the stacked transit signal stands above the noise. The detection threshold was **MES > 7.1σ** — everything in my dataset passed this threshold.

#### What MES tells you and does NOT tell you

**It tells you:** How confidently the periodic signal was detected. High MES = strong, repeatable signal. Low MES (near 7.1) = barely above threshold, could be noise.

**It does NOT tell you:** What the signal is. A massive eclipsing binary has an enormous MES (perhaps 1,000σ) — the signal is very clearly detected. But it is still a false positive. High MES is necessary but not sufficient for planethood.

#### Distribution by class
Confirmed planets: typically MES = 15–100 (many transits, moderate depth, cleanly detected)  
False positives: can range from 7 to 1,000+ (EBs with deep transits have very high MES)  
Candidates: typically MES = 7–20 (near detection threshold, marginal detections)

The MES distribution difference between classes is real but not as clean as the flag features — hence the moderate ANOVA score of 74.5.

---

### `koi_max_sngle_ev` — Maximum Single Event Statistic

**ANOVA F-Score: ~73**

#### What it is
The significance of the single strongest individual transit in the entire 4-year light curve:

```
SES_n = (depth_n × transit_duration) / (photometric noise over that duration)
```

The maximum over all n transits is `koi_max_sngle_ev`.

#### The diagnostic power — the ratio test

The real value of this feature comes from comparing it to MES:

```
For genuine planets:
All transits are approximately equal in depth and significance
SES_max ≈ MES / √N_transits
The ratio SES_max / (MES / √N) ≈ 1

For false positives from a single event:
One event dominates — cosmic ray hit, stellar flare, satellite crossing
SES_max >> MES / √N
The ratio SES_max / (MES / √N) >> 1
```

If one transit is enormously stronger than all the others, the signal may be dominated by a single anomalous event rather than a genuine repeating planet transit. This ratio is a subtle but real diagnostic — even though neither `koi_max_sngle_ev` nor `koi_max_mult_ev` individually has very high ANOVA scores, their combination is more informative.

> **My note:** This is why both are kept in the model despite their moderate ANOVA scores. The model may learn the implicit ratio between them — something a linear ANOVA test cannot detect.

---

### `koi_model_snr` — Transit Model Signal-to-Noise Ratio

**ANOVA F-Score: moderate**

#### What it is
The signal-to-noise ratio of the full fitted transit model:

```
SNR = transit depth / (photometric noise per unit depth)
     = how many sigma the best-fit transit stands above the noise
```

This is similar to MES but computed after full model fitting (which accounts for the transit shape, not just the box approximation). Slightly more precise than MES for characterised transits.

#### Relationship to other features
Highly correlated with MES. Both measure detection confidence. Both are in the model because they encode slightly different aspects of detection quality (BLS box approximation vs full transit model fit).

---

### `koi_num_transits` — Number of Observed Transits

**ANOVA F-Score: 279 — High**

#### What it is
How many times the planet transited during Kepler's 4-year observation window.

```
Approximate formula:
N_transits ≈ (4 years × 365.25 days/year) / koi_period (days)
           = 1,461 / period

Examples:
Period = 1 day:   N ≈ 1,461 transits
Period = 10 days: N ≈ 146 transits
Period = 100 days: N ≈ 14 transits
Period = 365 days: N ≈ 4 transits
Period = 500 days: N ≈ 2–3 transits
```

(Actual counts are slightly lower because Kepler had data gaps — module failures, safe mode events, quarterly rolls.)

#### Why it is one of the most important features

The number of observed transits is the **primary reason the CANDIDATE class exists.**

With many transits (N > 10):
- High combined MES → confident detection
- Can test for secondary eclipses (many even/odd pairs)
- Centroid analysis has high statistical power
- Odd-even depth difference test is reliable
- → Can confidently confirm OR rule out as false positive

With few transits (N < 5):
- Lower MES → less confident detection
- Almost impossible to detect secondary eclipse (only 2–3 odd/even pairs)
- Centroid shifts are harder to measure with sparse data
- Odd-even test has near-zero statistical power
- → Cannot confidently confirm OR rule out → CANDIDATE

This is why short-period planets are almost always either CONFIRMED or FALSE POSITIVE, while long-period planets are disproportionately CANDIDATE. The number of transits directly limits what diagnostic tests are possible.

ANOVA F-score of 279 reflects this strong connection: confirmed planets have systematically more transits than candidates, because more transits enable confirmation.

---

### `koi_bin_oedp_sig` — Odd-Even Depth Difference Significance

**ANOVA F-Score: moderate**

#### What it is
Tests whether odd-numbered transits (1st, 3rd, 5th...) and even-numbered transits (2nd, 4th, 6th...) have the same depth. Reports the statistical significance of any difference.

```
If odd-transit depth ≈ even-transit depth: koi_bin_oedp_sig ≈ 0  (planet-like)
If they differ significantly:               koi_bin_oedp_sig >> 0  (suspicious)
```

#### The physical reason this catches false positives

Here is the scenario it detects:

Imagine an eclipsing binary with a true orbital period of 10 days. Kepler's pipeline is trying every period in its search. At 5 days (half the true period), the pipeline sees:
- Odd-numbered events at 5 days: these are all the PRIMARY eclipses (deeper — larger star in front)
- Even-numbered events at 5 days: these are all the SECONDARY eclipses (shallower — smaller star in front)

The pipeline finds a "5-day period" with alternating deep and shallow transits. The alternating depth is the fingerprint of this period-halving mistake.

```
True signal (10-day EB):
Day 0:   primary eclipse (deep)   ▼▼▼
Day 5:   secondary eclipse (less) ▼
Day 10:  primary eclipse (deep)   ▼▼▼
Day 15:  secondary eclipse (less) ▼

Mistaken 5-day measurement:
Transit 1 (day 0):  ▼▼▼  (odd — deep)
Transit 2 (day 5):  ▼    (even — shallow)
Transit 3 (day 10): ▼▼▼  (odd — deep)
Transit 4 (day 15): ▼    (even — shallow)

koi_bin_oedp_sig would be very high — alternating depth is obvious
```

A genuine planet always produces the same depth (same planet, same star, same blocking area). No genuine planet mechanism creates alternating depths. So a high `koi_bin_oedp_sig` is a reliable false positive indicator.

#### Missing value note
This feature is missing for KOIs with very few transits. You need at least 2 transits of each parity to compute the test — meaning at least 4 transits total. KOIs with 1–3 transits (many candidates) have missing values here.

---

## Part 2 — Photometric Magnitude Features (7 features)

| Feature | Wavelength Band | Survey |
|---------|---------------|--------|
| `koi_kepmag` | Broad optical (420–900 nm) | Kepler |
| `koi_gmag` | Green (~480 nm) | SDSS |
| `koi_rmag` | Red (~620 nm) | SDSS |
| `koi_imag` | Near-infrared (~750 nm) | SDSS |
| `koi_zmag` | Far red (~900 nm) | SDSS |
| `koi_jmag` | Near-infrared (1.25 µm) | 2MASS |
| `koi_hmag` | Near-infrared (1.65 µm) | 2MASS |
| `koi_kmag` | Near-infrared (2.17 µm) | 2MASS |

---

### Why Multiple Wavelength Bands?

The **colour** of a star — how its brightness varies across different wavelengths — encodes physical information about the star and about potential contamination.

#### Understanding the magnitude scale

Lower magnitude number = **brighter** star. The scale is logarithmic:
```
Magnitude difference of 1 = brightness ratio of ~2.5×
Magnitude difference of 5 = brightness ratio of 100×

Sirius (brightest star in night sky): -1.5 mag
Typical Kepler target: 12–16 mag
Human eye limit: ~6 mag
```

#### Understanding star colours

Hot stars emit more blue light:
```
Hot star (F-type, 7,000 K): gmag ≈ rmag ≈ imag (roughly equal brightness in all optical bands)
Cool star (M-type, 3,500 K): gmag >> kmag (much brighter in infrared than blue)
```

The difference between magnitudes at different wavelengths is called a **colour index**:
```
(g - r) = koi_gmag - koi_rmag
(J - K) = koi_jmag - koi_kmag
```

Stellar type is encoded in these colour indices. Hot stars have blue colour indices (small g-r difference). Cool stars have red colour indices (large j-k difference).

---

### How Magnitudes Help Classify KOIs

#### Use 1 — Stellar Type Confirmation

The combination of all 7 magnitude bands allows reconstruction of the stellar spectral energy distribution (SED). This gives an independent estimate of stellar effective temperature without spectroscopy. Inconsistency between the photometric temperature and the spectroscopic temperature can indicate contamination or a misidentified stellar type.

#### Use 2 — Contamination Estimation

If a nearby star is present in the same Kepler pixel, its light dilutes the transit depth. The degree of dilution depends on the brightness difference between the target and the contaminant. Using magnitude data across multiple bands:

```
Dilution factor = L_target / (L_target + L_contaminator)

Transit depth_observed = transit depth_true × dilution factor
```

A background eclipsing binary that is 4 magnitudes fainter (100× fainter) dilutes a 50% eclipse into a 0.5% transit — planet-like in depth. Knowing the magnitudes of nearby stars (from surveys like 2MASS) allows estimation of this dilution factor.

#### Use 3 — Fainter Stars → More Uncertain Classification

As `koi_kepmag` increases (fainter star):
- More photometric noise → lower SNR → harder to detect
- Less spectroscopic follow-up → more missing values in stellar parameters
- Higher chance of background star contamination (fainter targets have denser backgrounds)
- Less likely to receive radial velocity confirmation

The magnitude features encode this "observational difficulty" signal that correlates with class membership.

---

### Why Seven Bands and Not Just One?

One magnitude tells you how bright the star is. Seven magnitudes at different wavelengths tell you what **kind** of star it is and how much contamination there might be.

The optical bands (g, r, i, z) and infrared bands (J, H, K) together cover a factor of ~4.5 in wavelength. A cool contaminating M-dwarf that is nearly invisible in the blue/green band becomes relatively more prominent in the infrared bands — detectable in the colour differences.

This multi-wavelength approach is standard in astronomical photometric classification. Including all seven bands allows the model to implicitly reconstruct these colour diagnostics without explicit feature engineering.

---

### ANOVA Context for Magnitude Features

Magnitude features individually have low-to-moderate ANOVA F-scores. They are not the dominant discriminators. Their value is:
- Contextual information that helps the model interpret other features
- Systematic bias detection (fainter targets → different false positive rates)
- Encoding contamination potential

They are more valuable in combination (the CNN can integrate information across all seven) than individually (which is all ANOVA tests).

---

## Summary — Signal Quality and Magnitude Group

| Feature | Why Included | Expected Model Use |
|---------|-------------|-------------------|
| `koi_max_mult_ev` | Overall detection confidence | High MES = clear signal but not necessarily a planet |
| `koi_max_sngle_ev` | Single-event vs. stacked consistency | Ratio to MES detects single-event false positives |
| `koi_model_snr` | Full model detection quality | Complements MES with model-fitted estimate |
| `koi_num_transits` | Statistical power of all tests | Low count → candidate; high count → confirmed or FP |
| `koi_bin_oedp_sig` | Period-halving EB detection | High value → period-halved eclipsing binary |
| Magnitudes (×7) | Stellar type and contamination | Context for interpreting transit parameters |

---

*Previous: [05 — False Positive Flag Features](05_feature_physics_fp_flags.md)*  
*Next: [07 — EDA: What I Explored and Why](07_eda_what_i_explored_and_why.md)*
