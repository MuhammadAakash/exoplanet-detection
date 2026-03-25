# 📚 Notes 02 — Data & Features

---

## What is Data?

Data is a giant spreadsheet where:
- Each **row** = one thing you're studying (one star)
- Each **column** = one measurement about that thing (one feature)
- The **last column** = the label (what you're trying to predict)

### Your Data Looks Like This:

| Star ID | Brightness Dip | Duration | Star Size | ... | Is Planet? |
|---|---|---|---|---|---|
| KOI-1 | 612 ppm | 2.95 hrs | 0.927 R☉ | ... | CONFIRMED |
| KOI-2 | 874 ppm | 4.50 hrs | 0.927 R☉ | ... | FALSE POSITIVE |
| KOI-3 | 488 ppm | 1.38 hrs | 1.200 R☉ | ... | CANDIDATE |

**3,900 rows × 37 features + 1 label = your dataset**

---

## What is a Feature?

A feature is any **measurable property** of the thing you're studying.

> **Analogy:** Describing a person to someone who has never met them.
> - Height, hair colour, age, glasses = **features**
> - "Is this person a doctor?" = **label**

The model learns: *"people with these feature combinations tend to be doctors"*

In your project: the model learns *"stars with these feature combinations tend to have real planets"*

---

## Why These 37 Features?

The original dataset had **141 columns**. We reduced to 37 by removing:

| What Was Removed | Why | How Many |
|---|---|---|
| Completely empty columns | No data at all — useless | 20 |
| Identifier columns | Names, dates, links — not predictive | 17 |
| Leaky columns | Derived from the same pipeline as labels — cheating | 2 |
| **Remaining** | **Genuine, non-leaky, predictive features** | **37** |

### What is Data Leakage?
Using information during training that wouldn't be available at prediction time — or that directly contains the answer.

> **Analogy:** A student who has the answer key hidden in their exam paper. They'd score 100% but learned nothing.

`koi_pdisposition` and `koi_score` were both outputs of NASA's own vetting pipeline — essentially pre-calculated answers. Using them would mean the model is just copying NASA's homework, not learning independently.

---

## The 37 Features — Complete Guide

### 🔵 Group 1: Transit Features (13 features)
*Describe the shape and behaviour of the brightness dip*

| Feature | Full Name | What It Measures | Why It Matters |
|---|---|---|---|
| `koi_period` | Orbital period | Days for one orbit | Real planets orbit consistently |
| `koi_time0bk` | Transit epoch | Time of first transit | Checks if transits repeat on schedule |
| `koi_impact` | Impact parameter | Where planet crosses star face (0=centre, 1=edge) | Affects dip shape |
| `koi_duration` | Transit duration | Hours the brightness dips | Too short/long suggests false positive |
| `koi_depth` | Transit depth | Brightness drop in ppm | Deeper = bigger planet |
| `koi_ror` | Radius ratio | Planet radius ÷ star radius | Very large = probably eclipsing binary |
| `koi_srho` | Stellar density | Density calculated from transit shape | Disagreement with measured density = fake transit |
| `koi_prad` | Planet radius | Size in Earth radii | >20 Earth radii = probably not a planet |
| `koi_sma` | Semi-major axis | Distance from star in AU | Determines temperature zone |
| `koi_incl` | Inclination | Angle of orbit (90° = edge-on = transit visible) | Low angle = no transit |
| `koi_teq` | Equilibrium temperature | Planet surface temperature in K | Reasonableness check |
| `koi_insol` | Insolation flux | Energy received vs Earth (1.0 = same as Earth) | Habitability indicator |
| `koi_dor` | Orbital distance/stellar radius | Orbit distance divided by star size | Small = very close orbit |

**Key unit: ppm = parts per million.** 
- 1% brightness drop = 10,000 ppm
- Earth-sized planet around Sun-sized star ≈ 84 ppm (tiny!)
- Jupiter-sized planet ≈ 10,000 ppm (1%)

---

### 🟣 Group 2: Stellar Features (8 features)
*Describe the star itself, not the planet*

| Feature | Full Name | What It Measures | Why It Matters |
|---|---|---|---|
| `koi_steff` | Effective temperature | Star surface temperature (K) | Sun = ~5,778K |
| `koi_slogg` | Surface gravity | Log of surface gravity | Giant stars (log g ~2) vs Sun-like (log g ~4.4) |
| `koi_smet` | Metallicity | Iron content vs Sun [Fe/H] | Metal-rich stars more likely to have planets |
| `koi_srad` | Stellar radius | Star size in Solar radii | Affects transit depth calculation |
| `koi_smass` | Stellar mass | Star mass in Solar masses | Used to calculate orbital distance |
| `ra` | Right Ascension | Sky coordinate (like longitude) | Star position for cross-referencing |
| `dec` | Declination | Sky coordinate (like latitude) | Star position for cross-referencing |
| `koi_kepmag` | Kepler magnitude | Star brightness through Kepler's detector | Brighter = cleaner measurements |

---

### 🔴 Group 3: False Positive Flags (4 features)
*Binary signals (0 or 1) that directly flag likely false positives*

These are among the **most powerful** features in the entire dataset.

| Feature | Flag Name | Meaning | What a "1" means |
|---|---|---|---|
| `koi_fpflag_nt` | Not Transit-Like | Signal doesn't look like a transit | Probably instrument noise or stellar pulsation |
| `koi_fpflag_ss` | Significant Secondary | Secondary eclipse detected | Almost certainly an eclipsing binary, not a planet |
| `koi_fpflag_co` | Centroid Offset | Light dip comes from a different location | Signal from background star, not target star |
| `koi_fpflag_ec` | Ephemeris Match | Timing matches a known false positive source | Contaminated by a known eclipsing binary nearby |

> **Why is `fpflag_ss` so powerful?**
> A real planet reflects very little light and produces no secondary eclipse. If you see a secondary dip when the "planet" goes behind the star, it means the "planet" is actually glowing — i.e. it's a star. Game over — false positive.

---

### 🟡 Group 4: Signal Quality Features (5 features)
*How strong and consistent is the detection?*

| Feature | Full Name | What It Measures | Why It Matters |
|---|---|---|---|
| `koi_max_sngle_ev` | Max Single Event Stat | Strongest single transit signal | High = one strong dip detected |
| `koi_max_mult_ev` | Max Multiple Event Stat (MES) | Signal averaged across ALL transits | Usually the **#1 most important feature** |
| `koi_model_snr` | Model SNR | Signal-to-noise of best-fit model | High SNR = signal stands out from noise |
| `koi_num_transits` | Number of transits | How many times planet crossed the star | More transits = more evidence |
| `koi_bin_oedp_sig` | Odd-Even Depth Difference | Difference between alternating transit depths | Eclipsing binaries have alternating depths |

> **MES explained simply:** Imagine you're listening for a pattern in a noisy room. If someone claps once, it might be random noise. If they clap 50 times at exactly the same interval, you're sure it's intentional. MES measures that consistency.

---

### 🟢 Group 5: Magnitude Features (7 features)
*Star brightness through different colour filters*

| Feature | Band | Wavelength | What It Shows |
|---|---|---|---|
| `koi_gmag` | g-band | Green optical | Blue-ish light from star |
| `koi_rmag` | r-band | Red optical | Red light from star |
| `koi_imag` | i-band | Near-IR optical | Near-infrared |
| `koi_zmag` | z-band | Far-red optical | Far-red light |
| `koi_jmag` | J-band | Near-infrared | Infrared from star |
| `koi_hmag` | H-band | Near-infrared | Infrared from star |
| `koi_kmag` | K-band | Near-infrared | Infrared from star |

Comparing brightness across bands reveals:
- The star's spectral type (what kind of star it is)
- Whether a contaminating background star is blended in

---

## The Golden Rule of Features

> **More features is NOT always better.**

Irrelevant or redundant features add noise and confuse the model.

> **Analogy:** Asking someone to identify a fruit by also telling them the weather that day. Irrelevant information just distracts.

Good feature selection = only giving the model information that **actually helps** it make the decision.

---

## Feature Groups Summary

```
37 Features
    │
    ├── Transit Features (13) ── Shape of the brightness dip
    │
    ├── Stellar Features (8)  ── Properties of the star itself
    │
    ├── FP Flags (4)          ── Direct false positive indicators ← Most powerful
    │
    ├── Signal Features (5)   ── Detection strength and consistency
    │
    └── Magnitude (7)         ── Multi-wavelength brightness measurements
```

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **Feature** | A measurable property used as input to the model |
| **Label** | The correct answer the model is trying to predict |
| **ppm** | Parts per million — unit for measuring tiny brightness drops |
| **Transit** | When a planet passes in front of its star, blocking some light |
| **False Positive** | Something that looks like a planet signal but isn't |
| **Eclipsing Binary** | Two stars orbiting each other — biggest source of false positives |
| **MES** | Multiple Event Statistic — measures how consistently a transit repeats |
| **SNR** | Signal-to-Noise Ratio — how strong the signal is compared to background noise |
| **Data Leakage** | Using information that cheats — answer already baked into the feature |
| **Overfitting** | Model memorises training data but fails on new data |

---

*Notes from MSc Data Science Dissertation — Exoplanet Candidate Vetting*
