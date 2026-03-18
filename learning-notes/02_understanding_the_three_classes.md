# 02 — Understanding the Three Classes

> These are the three "characters" in my dataset. Everything in the project comes back to understanding who they are, how they are different, and why the model sometimes confuses them.

---

## The Target Variable — `koi_disposition`

Every row in my dataset has been given a label by human experts after years of study. The label is stored in `koi_disposition` and has exactly three possible values.

**Dataset breakdown:**
```
CONFIRMED:      2,341 rows  (60.0%)  — definitely a planet
FALSE POSITIVE: 1,083 rows  (27.8%)  — definitely NOT a planet  
CANDIDATE:        477 rows  (12.2%)  — genuinely uncertain
```

---

## Class 1 — CONFIRMED (60%)

### Who they are

A CONFIRMED planet has been verified beyond reasonable doubt using independent evidence. The Kepler team has ruled out all plausible false positive scenarios and positively identified the object as a planet.

### How confirmation happens

Confirmation is not automatic — it requires additional work beyond the Kepler data:

**Radial velocity:** A planet's gravity tugs its host star as it orbits. This causes tiny Doppler shifts in the star's spectrum — the starlight shifts slightly blue when the planet pulls the star toward us, and slightly red when the planet pulls away. Measuring this wobble gives you the planet's mass. If the mass is below ~13 Jupiter masses, it is a planet. This requires a large ground-based spectrograph, typically on a 4–10 metre telescope.

**Transit Timing Variations (TTV):** When multiple planets orbit the same star, they gravitationally interact and pull each other's orbits slightly. This causes transits to arrive slightly earlier or later than predicted. The pattern of timing variations uniquely fingerprints a multi-planet system. No false positive mechanism creates this pattern.

**Statistical validation:** For faint stars where radial velocity is impossible, Bayesian statistical tools (Vespa, BLENDER) calculate the probability that the signal is a planet versus every plausible false positive scenario. If the planet probability exceeds 99.73% (3 sigma), the KOI is statistically validated as a planet — even without a mass measurement.

### Typical characteristics in the dataset

- `koi_depth` < 10,000 ppm (planet-sized dip, not stellar-sized)
- `koi_prad` < 15 Earth radii (physically possible planet size)
- All four FP flags = 0 (no false positive signatures detected)
- `koi_num_transits` relatively high (well-observed, many transits stacked)
- `koi_score` near 1.0 (Robovetter agrees: this is a planet)
- `koi_impact` mostly below 0.8 (not grazing transits)

### Why 60% of the dataset is confirmed

Kepler ran for 4 years, and the community had years afterward to confirm candidates. Many of the "easy" confirmations — bright stars, short periods, large planets — were confirmed first. By the time the DR25 catalogue was released, most clearly planet-like signals had been confirmed. What remains as FALSE POSITIVE or CANDIDATE is systematically harder.

---

## Class 2 — FALSE POSITIVE (27.8%)

### Who they are

FALSE POSITIVE KOIs are signals that initially looked like planets but were revealed to be something else. They passed Kepler's automated detection threshold (MES > 7.1) but failed subsequent vetting.

### The main types of false positives

**Type 1 — Eclipsing Binary (EB):**  
Two stars orbiting each other. When one passes in front of the other, it creates a brightness dip that mimics a planet transit. The giveaways:
- Dip is much deeper than any planet could produce
- Second dip occurs exactly halfway through the orbit (the other star passing behind)
- Transit shape is often V-shaped rather than flat-bottomed (grazing geometry)

```
Planet transit:          Eclipsing binary transit:
    ╔════╗                       ╱╲
════╝    ╚════            ══════╱  ╲══════
(flat bottom = solid disc)   (V-shape = stellar limb crossing)
```

**Type 2 — Background Eclipsing Binary (BEB):**  
A faint eclipsing binary star that happens to be in the same Kepler pixel as the target star. Kepler's pixels are ~4 arcseconds wide — large enough to contain multiple stars. The EB's brightness variations get diluted by the bright target star, creating a shallow signal that looks like a planet transit.

The giveaway: during the transit, the **centroid** (centre of light) shifts slightly toward the background EB. The target star's centroid should be rock-steady if the transit is on the target. Any shift means the signal is coming from somewhere else.

**Type 3 — Hierarchical Triple:**  
A distant third star in the same system as a close stellar pair. The third star's eclipses are diluted by the bright pair, appearing shallow and planet-like. Very hard to detect without high-resolution imaging.

**Type 4 — Grazing Eclipsing Binary:**  
Two stars where one barely clips the edge of the other during their eclipse. The transit is V-shaped (never achieves full depth) and very short. The `koi_fpflag_nt` (not transit-like) flag is designed specifically to catch this.

**Type 5 — Instrumental / Background:**  
Systematic detector noise, cosmic rays, or optical artefacts on the Kepler CCD creating spurious periodic signals.

### Typical characteristics in the dataset

- `koi_depth` often > 50,000 ppm (stellar-sized dip)
- `koi_prad` often > 15 Earth radii (physically impossible for a planet)
- One or more FP flags = 1 (failed at least one diagnostic test)
- `koi_score` near 0.0 (Robovetter agrees: this is not a planet)
- `koi_bin_oedp_sig` sometimes elevated (alternating depth in odd/even transits)

### Why outliers in this class are NOT errors

The most extreme values in the entire dataset — planet radii of 100, 200, even 300 Earth radii — all belong to false positives. These are NOT errors in the data. They are correctly computed values for what happens when the Kepler pipeline fits a "planet" model to an eclipsing binary signal.

The pipeline assumes it is measuring a planet. For an EB, the "planet radius" it computes is actually the secondary star's radius — which can be stellar in size. **Removing these outliers would remove the most extreme and clearly identifiable false positives from training data.** They must be kept.

---

## Class 3 — CANDIDATE (12.2%)

### Who they are

CANDIDATE KOIs are the genuinely uncertain ones. The signal passes all automated vetting tests — it is not obviously a false positive — but has not received enough follow-up observation to confirm or rule out as a planet.

**Important:** CANDIDATE does not mean "probably a planet." Some candidates will turn out to be confirmed planets. Others will turn out to be false positives that were not caught by automated methods. The model's job for candidates is to assess probability, not declare certainty.

### Why they remain uncertain

**Reason 1 — Too few transits:**  
A planet with a 200-day orbital period only transited Kepler about 7 times in 4 years. With only 7 transits, statistical tests have low power. You cannot reliably test for secondary eclipses, centroid offsets, or odd-even depth differences with so few data points.

**Reason 2 — Faint host star:**  
Radial velocity measurements require enough starlight to take a high-resolution spectrum. Stars fainter than about magnitude 15 are too faint for most spectrographs. Many CANDIDATE host stars are simply too faint to confirm.

**Reason 3 — Habitable zone orbits:**  
Planets in the habitable zone (orbital period ~200–400 days) have only 3–7 transits in Kepler's dataset. These are scientifically the most important candidates — potentially Earth-like planets in the right temperature zone — but also the hardest to confirm.

**Reason 4 — Ambiguous geometry:**  
Some signals have parameter combinations that do not clearly rule out false positives but are not obviously EB-like either. These sit in a grey zone where more data would help but has not been obtained.

### Typical characteristics in the dataset

- `koi_num_transits` = 1–10 (too few for reliable statistical tests)
- All four FP flags = 0 (no obvious false positive signatures found)
- `koi_score` spread across 0.4–0.9 (automated uncertainty reflects genuine uncertainty)
- `koi_depth` and `koi_prad` overlap with both other classes (genuinely ambiguous)
- Missing values in more diagnostic features (fewer transits = fewer tests computable)

### Why CANDIDATE is the hardest class to predict

The CANDIDATE class genuinely overlaps with both CONFIRMED and FALSE POSITIVE in feature space. This is not a failure of the model — it reflects the genuine scientific uncertainty. A model that predicts CANDIDATE with high confidence is correctly capturing the fact that the signal is ambiguous.

This is also why CANDIDATE will have the lowest per-class F1 score in all my models. That is **correct behaviour**, not a problem to fix.

---

## The Label Encoding

For modelling, I encode the three classes as integers:

```python
CONFIRMED      → 0
FALSE POSITIVE → 1
CANDIDATE      → 2
```

The choice of numbers is arbitrary — the model does not treat 0 as "better" than 2. Any three distinct integers would work.

---

## Why Class Imbalance Matters

```
CONFIRMED:      2,341  ████████████████████████████████████████ 60%
FALSE POSITIVE: 1,083  ████████████████████ 27.8%
CANDIDATE:        477  █████████ 12.2%

Imbalance ratio: 4.9× (CONFIRMED to CANDIDATE)
```

A model that simply predicts CONFIRMED for every input achieves **60% accuracy without learning a single pattern.** This is called the majority class baseline. It is useless in practice but looks impressive on paper.

This is why:
- I use `class_weight='balanced'` — forces the model to pay equal attention to all three classes
- I report **macro F1, Cohen's Kappa, and MCC** — these penalise models that ignore minority classes
- I use **stratified splits** — every train/val/test set has the same 60/28/12 ratio

---

## What the Model Needs to Learn

| Task | Difficulty | Why |
|------|------------|-----|
| Separating CONFIRMED from FALSE POSITIVE | Medium | FP flags are very strong signals |
| Separating FALSE POSITIVE from CANDIDATE | Medium | FP class has clear flag signatures |
| Separating CONFIRMED from CANDIDATE | Hard | Both have clean flags; differ mainly in confidence and number of transits |
| Correctly labelling ambiguous CANDIDATEs | Very hard | Genuinely uncertain by definition |

---

*Previous: [01 — How Kepler Finds Planets](01_how_kepler_finds_planets.md)*  
*Next: [03 — Transit Geometry Features](03_feature_physics_transit.md)*
