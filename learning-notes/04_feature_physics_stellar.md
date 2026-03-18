# 04 — Feature Physics: Stellar Parameter Features

> These 8 features describe the **host star**, not the planet. They matter because every derived planet measurement depends on knowing the stellar parameters correctly. A wrong star = wrong planet.

---

## The 8 Stellar Features at a Glance

| Feature | What It Measures | Unit |
|---------|-----------------|------|
| `koi_steff` | Star surface temperature | Kelvin |
| `koi_slogg` | Star surface gravity | log₁₀(cm/s²) |
| `koi_smet` | Star iron abundance relative to Sun | dex [Fe/H] |
| `koi_srad` | Star physical radius | Solar radii |
| `koi_smass` | Star physical mass | Solar masses |
| `ra` | Sky position (east-west) | degrees |
| `dec` | Sky position (north-south) | degrees |
| `koi_kepmag` | Star brightness in Kepler band | magnitude |

---

## `koi_steff` — Stellar Effective Temperature

### What it is
The temperature of the stellar **photosphere** — the visible surface layer from which light escapes. Not the core temperature (which is millions of Kelvin), but the surface we can see and measure.

Measured in Kelvin. Derived from spectroscopy (analysing the spectrum of light from the star) or from photometric colour (how the star's brightness ratio varies across different wavelength bands).

### Stellar types and their temperatures
```
Spectral  Colour    Temperature    Characteristics
Type      
────────────────────────────────────────────────────────────────
O         Blue      30,000–60,000 K  Rare, very massive, short-lived
B         Blue      10,000–30,000 K  Bright, pulsate strongly
A         White     7,500–10,000 K   Rapidly rotating
F         Yellow    6,000–7,500 K    Slightly hotter than Sun
G         Yellow    5,200–6,000 K    ← Sun is here (5,778 K)
K         Orange    3,700–5,200 K    Smaller, calmer than Sun
M         Red       2,400–3,700 K    Most common star type in galaxy
```

Kepler targeted mostly **FGK stars** — solar-type stars — because they were thought most likely to host Earth-like planets and are photometrically stable enough to detect small transits.

### Why it matters for classification
- **M-dwarf hosts (Teff < 4,000 K):** Small stars → any given planet produces a larger transit depth (depth = (Rp/Rs)² — small Rs means large depth for same Rp). Changes the depth-to-size relationship significantly. Easier to detect but different parameter ranges than FGK targets.
- **F-star hosts (Teff > 6,200 K):** Hot stars pulsate and have stronger stellar activity. This intrinsic variability adds noise that can mimic or mask transit signals.
- **Very hot or very cool stars:** Were deprioritised in Kepler's target selection, so their KOIs have fewer follow-up observations → more likely to remain as candidates.

ANOVA F-score: **91** — moderate. Different stellar populations have systematically different false positive rates.

### Where the data comes from
Most stellar temperatures in the KOI catalogue come from the **Kepler Input Catalogue (KIC)** photometric estimates or subsequent spectroscopic surveys (LAMOST, SpecMatch, CKS). The KIC photometric temperatures have systematic errors of ~200 K and were revised significantly in later data releases.

---

## `koi_slogg` — Stellar Surface Gravity

### What it is
The gravitational acceleration at the star's surface, expressed as a **base-10 logarithm**:

```
log g = log₁₀(G × M★ / R★²)

Where:
G  = gravitational constant = 6.674 × 10⁻⁸ cm³ g⁻¹ s⁻²
M★ = stellar mass in grams
R★ = stellar radius in centimetres
```

The result is in log₁₀(cm/s²). The Sun has log g = 4.44.

### Typical values by stellar evolutionary stage
```
Star Type              log g      Physical gravity
────────────────────────────────────────────────────────────────
Main sequence dwarfs   4.0–4.5    10,000–32,000 cm/s² (Earth = 980)
Subgiants              3.5–4.0    3,000–10,000 cm/s²
Red giants             2.0–3.5    100–3,000 cm/s²
White dwarfs           ~8.0       100 million cm/s²
```

### Why it is critical — the evolved star false positive

This is one of the most important features for catching a specific type of false positive.

Here is the scenario: The Kepler Input Catalogue (KIC) was assembled before Kepler launched using photometric colours alone. Photometric classification can misidentify evolved stars (subgiants, giants) as main-sequence dwarfs, because their colours can be similar.

If a giant star (radius 5–50 × Solar radius) was misclassified as a dwarf in the KIC, then:
- The derived planet radius uses the wrong (too small) stellar radius
- A "planet radius" of 2 Earth radii from the depth measurement could actually be a companion star
- An eclipsing binary around a giant looks like a planet transit around a dwarf

The giveaway: `koi_slogg < 3.5` indicates an evolved star. Any KOI hosted by a star with log g < 3.5 deserves extra scrutiny. This feature encodes that scrutiny.

### How it is measured
Surface gravity requires either:
- **Spectroscopy:** High-resolution spectrum fit with stellar atmosphere models (most accurate)
- **Isochrone fitting:** Using temperature, luminosity, and stellar evolution models
- **Asteroseismology:** Some Kepler targets were bright enough to detect stellar oscillations, which directly give density and surface gravity

---

## `koi_smet` — Stellar Metallicity [Fe/H]

### What it is
The abundance of iron in the star's atmosphere relative to the Sun, in logarithmic scale:

```
[Fe/H] = log₁₀(N_Fe/N_H)★ - log₁₀(N_Fe/N_H)☉

[Fe/H] = 0.0:   Exactly Solar metallicity
[Fe/H] = +0.3:  Twice the Solar iron abundance
[Fe/H] = -0.3:  Half the Solar iron abundance
[Fe/H] = +0.5:  Three times Solar (very metal-rich)
[Fe/H] = -1.0:  One-tenth Solar (old, metal-poor halo star)
```

Iron is used as the reference element because it has hundreds of easily measurable spectral lines.

### The planet-metallicity correlation — one of the most robust findings in exoplanet science
Metal-rich stars are significantly more likely to host giant planets:
- Stars with [Fe/H] > +0.3: ~25% probability of having a giant planet
- Stars with [Fe/H] ≈ 0: ~3% probability
- Stars with [Fe/H] < -0.3: < 1% probability

The physical reason: protoplanetary discs around metal-rich stars contain more solid material (rocks, ices) to build planetary cores. Without a solid core, giant planets cannot form.

This means the **confirmed planet population should show a mild metallicity enhancement** compared to false positives (which are star-star systems, not subject to the same bias). The ANOVA F-score of 161 reflects this real but moderate signal.

### The missing data problem
Approximately **30% of koi_smet values are missing.** The reason is physically motivated: accurate metallicity requires a high-resolution spectrum, which requires enough starlight for the spectrograph. Stars fainter than about Kepler magnitude 15 are too faint for most spectrographs.

This means missingness in `koi_smet` is **not random** — it correlates with star brightness and therefore with detection quality. Stars without metallicity measurements are systematically fainter and harder to characterise. This is a missing-not-at-random (MNAR) pattern, which is the most complex type of missingness to handle correctly.

> **My note:** I impute missing metallicity with the **median** of the training set metallicity values. This is imperfect — the missing cases are systematically different from those with measurements. But it is better than dropping those rows (losing too many training examples) or using mean (pulled toward metal-rich outliers). Acknowledging this limitation in the dissertation is important.

---

## `koi_srad` — Stellar Radius (Solar Radii)

### What it is
The physical size of the host star. Sun = 1.0 R☉ = 696,000 km.

```
Typical values:
M-dwarf: 0.1–0.6 R☉    (smallest, faintest)
K-dwarf: 0.6–0.9 R☉
G-dwarf (Sun): 0.85–1.1 R☉
F-dwarf: 1.1–1.5 R☉
A-star:  1.5–2.5 R☉
Subgiant: 1.5–5 R☉
Giant:  5–50 R☉
```

### Why it is critical — the transit-to-planet-size conversion
Every planet size measurement depends directly on the stellar radius:

```
Rp = koi_ror × koi_srad × (R☉ / R⊕)
   = koi_ror × koi_srad × 109.2
```

A 10% error in stellar radius propagates directly into a 10% error in planet radius. A 50% error (which was common for giant/dwarf misclassification in the KIC) means planet sizes were systematically wrong for those targets.

### Correlation with other features
`koi_srad` is highly correlated with `koi_steff` (hotter stars are larger on the main sequence) and anti-correlated with `koi_slogg` (larger stars have lower surface gravity). These are physically expected correlations reflecting the Hertzsprung-Russell diagram.

---

## `koi_smass` — Stellar Mass (Solar Masses)

### What it is
Physical mass of the host star. Sun = 1.0 M☉ = 1.989 × 10³⁰ kg.

```
M-dwarf: 0.08–0.6 M☉
K-dwarf: 0.6–0.9 M☉
G-dwarf (Sun): 0.85–1.1 M☉
F-dwarf: 1.1–1.5 M☉
A-star:  1.5–3 M☉
```

### Why it matters
1. **Kepler's Third Law:** Converting period to orbital distance requires stellar mass: `a = [GM★ T²/4π²]^(1/3)`
2. **Habitable zone location:** More massive stars are more luminous, shifting the habitable zone outward
3. **Planet formation:** More massive stars have more massive protoplanetary discs, affecting what kinds of planets form

### Measurement
Stellar mass is not directly measured — it is derived from spectroscopic temperature and surface gravity using stellar evolution models (isochrones). The precision is typically ±5–10%.

---

## `ra` and `dec` — Sky Coordinates

### What they are
The celestial equivalent of longitude and latitude on the sky:
- **Right Ascension (RA):** East-west position, measured in degrees (0°–360°)
- **Declination (Dec):** North-south position, measured in degrees (-90° to +90°)

These tell you exactly where on the sky the star is located.

### Why they are in a machine learning model — the interesting reason

At first glance, sky coordinates seem irrelevant to classifying a planet. But there are real systematic effects encoded in these coordinates:

**Kepler's detector structure:** Kepler had 42 CCD modules arranged in a fixed pattern. Stars in different parts of the field of view hit different parts of the detector. Some modules had better photometry; some had more systematic noise. Stars near module edges experienced more data gaps. These detector-position effects are encoded in RA and Dec.

**Background contamination density:** Some directions on the sky have denser stellar backgrounds (towards the galactic centre, for instance). Denser backgrounds mean higher probability that a faint eclipsing binary is lurking in the same pixel as the target — increasing the false positive rate from that sky direction.

**Bright star contamination:** Very bright stars bleed charge across detector columns regardless of sky position, but the bleeding direction depends on detector orientation (encoded in coordinates).

The model can learn these systematic effects from coordinates alone. Including RA and Dec allows the model to correct for detector-position and sky-density biases automatically.

ANOVA F-scores for RA and Dec are moderate — they do not individually predict class membership strongly, but they provide useful contextual information about the observational environment.

---

## `koi_kepmag` — Kepler Magnitude

### What it is
The apparent brightness of the host star as seen through Kepler's broad optical filter (420–900 nm bandpass). The magnitude scale is logarithmic and inverted: **lower number = brighter star.**

```
Magnitude scale reference:
Sirius (brightest star in night sky): -1.5
Typical Kepler target: 12–16
Faintest Kepler targets: ~17
Human eye limit: ~6
```

The difference of 5 magnitudes = factor of 100 in brightness.

### Why it matters for classification

**Photometric precision:** Brighter stars give more photons per 30-minute cadence → lower Poisson noise → higher SNR per transit → easier to detect and characterise the transit.

**Follow-up feasibility:** Brighter stars (kepmag < 13) can receive radial velocity measurements with standard spectrographs → easier to confirm or rule out as false positives → more likely to be CONFIRMED.

**Background contamination:** Fainter stars (kepmag > 15) are more likely to have a nearby bright background star in the same pixel. This dilutes transit depths and creates ambiguous situations.

**Missing stellar parameters:** Stars with kepmag > 15.5 rarely received spectroscopic follow-up → missing values in `koi_smet`, `koi_slogg`, `koi_srad` are correlated with high kepmag.

---

## The Dependency Chain — Why Stellar Features Are Critical

Every planet measurement flows through the stellar parameters:

```
Light curve data
       ↓
Transit model fitting
       ↓
koi_ror (model-independent — from depth alone)
       ↓
× koi_srad (stellar radius — from spectroscopy or KIC)
       ↓
koi_prad (physical planet radius in Earth radii)

And:
koi_period
       ↓
× f(koi_smass) [Kepler's Third Law]
       ↓
koi_sma (orbital distance in AU)
       ↓
koi_teq (equilibrium temperature)
```

If any stellar parameter is wrong, all downstream derived quantities are wrong. This is why the KIC misclassification problem (dwarfs mislabelled as giants) was such a significant source of false positives in early Kepler analyses.

---

*Previous: [03 — Transit Geometry Features](03_feature_physics_transit.md)*  
*Next: [05 — False Positive Flag Features](05_feature_physics_fp_flags.md)*
