# 03 — Feature Physics: Transit Geometry Features

> These 13 features describe the **shape, size, timing, and geometry** of the brightness dip itself. They come from fitting a mathematical transit model to the Kepler light curve.

---

## The 13 Transit Features at a Glance

| Feature | What It Measures | Unit |
|---------|-----------------|------|
| `koi_period` | How long one orbit takes | days |
| `koi_time0bk` | When the first transit occurred | days (BKJD) |
| `koi_impact` | How centrally the planet crossed the star | dimensionless (0–1) |
| `koi_duration` | How long the transit lasted | hours |
| `koi_depth` | How much the star dimmed | parts per million |
| `koi_ror` | Planet radius / star radius | dimensionless |
| `koi_srho` | Stellar density from transit shape | g/cm³ |
| `koi_prad` | Physical planet radius | Earth radii |
| `koi_sma` | Average orbital distance | AU |
| `koi_incl` | Tilt of orbit relative to sky plane | degrees |
| `koi_teq` | Planet's equilibrium temperature | Kelvin |
| `koi_insol` | Stellar radiation received by planet | Earth flux units |
| `koi_dor` | Orbital distance / star radius | dimensionless |

---

## `koi_period` — Orbital Period

### What it is
How long it takes the planet to complete one full orbit around its star. Measured in Earth days.

### How it is measured
The Box Least Squares algorithm searches millions of trial periods. The correct period is the one where all transits line up perfectly when you "fold" the light curve. Once found, the period is known to extraordinary precision — often better than 1 second for short-period planets.

### Physical ranges in dataset
- Shortest: ~0.5 days (ultra-hot Jupiter, skimming the stellar surface)
- Longest: ~500 days (Earth-like planet at 1 AU)
- Most common for confirmed: 1–100 days

### Connection to orbital distance — Kepler's Third Law
```
T² ∝ a³  (simplified for T in years, a in AU)

More precisely:
T² = (4π² / GM★) × a³

Where:
T  = period
a  = semi-major axis (orbital distance)
M★ = stellar mass
G  = gravitational constant
```
Knowing the period and the stellar mass gives you the orbital distance exactly. This is why `koi_period` and `koi_sma` are highly correlated (~0.85).

### Why it matters for classification
- Very short periods (< 1 day): Physically suspicious — the planet would be inside or near the star's surface. Could be a contact binary (two touching stars). Likely false positive.
- Long periods (> 300 days): Only 3–5 transits in Kepler's 4-year window. Too few transits to confirm statistically. Likely candidate.
- Confirmed planets: Cluster at 1–100 days (easy to detect, many transits accumulated).

> **My note:** The period distribution directly encodes an observational bias — Kepler was better at finding close-in planets simply because they transit more often. This is selection bias baked into the dataset. Short-period planets are overrepresented among confirmed planets not because they are more common in the universe, but because they are easier to detect.

---

## `koi_time0bk` — Time of First Transit (BKJD)

### What it is
The timestamp of the first observed transit mid-point. Measured in Barycentric Kepler Julian Days (BKJD), which is the standard Julian Day minus 2,454,833.0.

### Why it is in the model
Mostly for reproducibility and phase-folding calculations. Not a strong classifier by itself. Its discriminating power comes from being an anchor for all timing-related diagnostics. Suspicious timing variations (transits arriving earlier or later than predicted) can indicate gravitational interactions or contamination.

### ANOVA relevance
Low individual discriminating power. Included for completeness and model correctness, not as a primary signal.

---

## `koi_impact` — Impact Parameter

### What it is
The sky-projected distance between the planet's transit chord and the centre of the stellar disc, normalised by the stellar radius.

```
b = (a / Rs) × cos(i)

Where:
a  = semi-major axis
Rs = stellar radius  
i  = orbital inclination
```

### Physical meaning — visually
```
Star (viewed from above during transit):

b = 0.0: planet crosses dead centre
    ○○○○○○○○
    ○       ○
    ○───────○  ← transit chord through centre
    ○       ○
    ○○○○○○○○

b = 0.5: planet crosses halfway between centre and edge
    ○○○○○○○○
    ○       ○
    ○       ○
    ○───────○  ← transit chord
    ○○○○○○○○

b = 0.9: planet barely grazes the stellar limb (grazing transit)
    ○○○○○○○○
    ○       ○
    ○       ○
    ○       ○
    ─────────  ← transit chord barely crosses edge
```

### Effect on transit shape
- Low b (< 0.5): Flat-bottomed U-shape. Long transit. Deep and clear.
- High b (0.7–0.9): Shallower, shorter, more triangular/V-shaped transit.
- b > 1.0: No transit occurs at all.

### Why it matters for classification
High impact parameters (b > 0.8) produce **grazing transits** — V-shaped rather than U-shaped. Grazing eclipsing binaries almost always have high impact parameters. The `koi_fpflag_nt` flag is essentially detecting the V-shape that high-b systems produce.

ANOVA F-score: **505** — 4th highest in dataset. This surprised me initially. The reason is that false positives systematically have higher impact parameters (grazing geometry is common for EBs) while confirmed planets have a more uniform distribution of b values.

---

## `koi_duration` — Transit Duration T14

### What it is
Time from first contact (planet's leading edge touches stellar limb) to fourth contact (trailing edge leaves stellar limb). Called T14 because it spans from contact 1 to contact 4.

Measured in hours.

### The physics of duration
```
Simplified:
T14 ≈ √[(Rs + Rp)² - (b·Rs)²] × (2/v_orbital)

Where v_orbital = orbital velocity = 2πa/T

More precisely:
T14 = (P/π) × arcsin(√[(Rs+Rp)² - (b·Rs)²] / a)
```

Duration depends on:
- **Stellar size (Rs):** Bigger star → longer transit
- **Impact parameter (b):** Higher b → shorter transit (shorter chord)
- **Orbital velocity:** Faster orbit → shorter transit. Faster orbit = closer-in planet.
- **Planet size (Rp):** Larger planet adds slightly more to chord length

### Why it matters for classification
Duration combined with period can reveal inconsistencies. A very long duration for a very short period implies an orbital velocity inconsistent with the stellar density — a sign that something is wrong with the assumed geometry (common for background EBs where the geometry belongs to a different star).

---

## `koi_depth` — Transit Depth

### What it is
How much the star dimmed during transit. Measured in parts per million (ppm).

```
depth = (Rp / Rs)² × 10⁶   [in ppm]

Or equivalently:
depth = (koi_ror)²  × 10⁶
```

### Scale of depths in real situations
```
Object                          Depth
──────────────────────────────────────────
Earth transiting Sun            84 ppm
Neptune transiting Sun          1,250 ppm
Jupiter transiting Sun          10,000 ppm
Jupiter-sized planet            ~10,000 ppm
Brown dwarf companion           ~50,000–200,000 ppm
M-dwarf eclipsing G-star        ~100,000–400,000 ppm
Equal-mass eclipsing binary     up to 500,000 ppm (50% dimming)
```

### Why it is one of the most powerful features
The difference between a planetary transit and an eclipsing binary is fundamentally a difference in **scale**. This scale difference shows up directly in depth:
- Planet depth: < ~10,000 ppm
- Stellar eclipse: > ~50,000 ppm (usually much more)
- Grey zone: 10,000–50,000 ppm (could be a large planet or a diluted EB)

The values above 50,000 ppm in my dataset are not errors. They are eclipsing binaries whose implied "planet" size is impossible for a real planet. The model must learn this hard physical boundary.

### EDA finding
Depth shows clear class separation. False positives cluster at high depths; confirmed planets cluster at low-to-moderate depths; candidates span the middle range. This pattern was visible in both the distribution plots (Section 3) and the period-depth scatter plot (Section 5).

---

## `koi_ror` — Planet-to-Star Radius Ratio (Rp/Rs)

### What it is
The ratio of planet radius to stellar radius. Directly derived from transit depth:
```
koi_ror = Rp/Rs = √(depth / 10⁶) = √(koi_depth / 1,000,000)
```

### Why it exists separately from `koi_depth`
`koi_ror` is the **model-independent** measurement — it comes directly from the light curve shape without needing to know anything about the star. It is the most directly "observed" quantity.

`koi_depth` is the same thing expressed as ppm (just `koi_ror²` scaled). Both are in the model because they are functionally equivalent but the model can learn from either form depending on what combination helps.

### Correlation with depth
r ≈ 0.90 (expected — they are mathematically linked). Both appear as informative features because they encode the same physical information from slightly different mathematical angles.

---

## `koi_srho` — Stellar Density from Transit Fit

### What it is
This is one of the most powerful and conceptually elegant features in the dataset.

The **shape** of the transit — specifically the ratio of the ingress duration (T12, the sloped entry) to the total transit duration (T14) — mathematically encodes the mean density of the host star:

```
ρ★ ∝ (T14² - T23²)^(3/2) / (P × T14³)

Where:
T14 = total transit duration (first to fourth contact)
T23 = flat-bottom duration (second to third contact)
P   = orbital period
```

### The rho-star test — why this is a false positive diagnostic
Here is the key insight: there are **two independent ways** to measure stellar density for a Kepler host star:

1. **From the transit:** Using the equation above with transit timing measurements
2. **From spectroscopy:** Measuring the star's spectrum gives temperature and surface gravity, from which density follows directly

If the star is what it appears to be and the transit is genuine, these two measurements must agree. If they disagree significantly, something is wrong. 

The most common cause of disagreement: the transit is coming from a **background eclipsing binary**. The transit geometry belongs to the background EB (which has different stellar density), but the spectroscopic density was measured from the target star. The mismatch is the fingerprint of contamination.

This disagreement is called the **rho-star test** or **stellar density test** and is one of the most rigorous automated false positive diagnostics.

---

## `koi_prad` — Planet Radius (Earth Radii)

### What it is
The physical size of the planet, derived by combining the geometric ratio (koi_ror) with the stellar size (koi_srad):

```
koi_prad = koi_ror × koi_srad × (R☉ / R⊕)
         = koi_ror × koi_srad × 109.2

Where:
R☉ = Solar radius
R⊕ = Earth radius
koi_srad = stellar radius in Solar radii
```

### The physical boundaries
```
Object Class          Radius (R⊕)
──────────────────────────────────
Rocky planets         0.5 – 1.5
Super-Earths          1.5 – 2.0
Mini-Neptunes         2.0 – 4.0
Neptunes              3.9 – 4.0
Sub-Saturns           4.0 – 8.0
Jupiters              8.0 – 11.2
──────────────────────────────────
PHYSICAL LIMIT FOR PLANETS: ~11.2 R⊕ (1 Jupiter radius)
──────────────────────────────────
Brown dwarfs          11.2 – 80 R⊕
Stars (M-dwarfs)      80+ R⊕
```

Nothing larger than ~11–15 Earth radii can be a planet. Objects above this size are either brown dwarfs or stars. This hard physical boundary is one of the clearest false positive indicators in the dataset.

### The Fulton Gap
There is a known deficit of planets at 1.5–2.0 Earth radii, called the **Fulton Gap** (discovered from Kepler data in 2017). This gap separates rocky super-Earths from gas-rich mini-Neptunes. Confirmed planets show this gap in their radius distribution. False positives do not — they scatter at all sizes including physically impossible ones.

### Why both koi_prad and koi_ror are in the model
- `koi_ror` is more reliable (does not depend on knowing stellar radius)
- `koi_prad` is more physically interpretable (can compare to physical size limits)
- High correlation (r ≈ 0.95) but each carries slightly different information

---

## `koi_sma` — Semi-Major Axis (AU)

### What it is
The average orbital distance between planet and star. 1 AU = Earth-Sun distance = 149.6 million km.

Derived from period using Kepler's Third Law:
```
a = [GM★/(4π²) × T²]^(1/3)

In convenient units: a (AU) ≈ [M★/M☉ × (T/365.25)²]^(1/3)
```

### Correlation with period
r ≈ 0.85 — because `a` is directly derived from `T` via Kepler's Third Law.

### Correlation with equilibrium temperature
r ≈ -0.85 — because planets closer to their stars are hotter:
```
Teq ∝ a^(-1/2)
```
These physically motivated correlations appeared exactly as expected in the correlation heatmap (EDA Section 4).

---

## `koi_incl` — Orbital Inclination (Degrees)

### What it is
The angle between the planet's orbital plane and the plane perpendicular to our line of sight. 90° means perfectly edge-on (we see the transit). Less than 90° means the orbit is tilted away from us.

```
                  Our line of sight
                         ↑
                         │
         i = 90°:    ────│──── orbital plane (edge-on, transit occurs)
                         │

         i = 70°:    ────/──── orbital plane (tilted, transit may not occur)
```

### Only edge-on planets are detectable
For a transit to occur, the inclination must satisfy:
```
cos(i) < Rs/a   (roughly)
```
For most Kepler planets, this requires i to be within a fraction of a degree of 90°.

### Why inclination discriminates despite this
Since all detected planets must have i ≈ 90°, inclination should not vary much in a clean planet sample. Departures from 90° mean the model-fitting behaved unusually — which happens for:
- Grazing eclipsing binaries (transit geometry is poorly constrained)
- Background EBs (the transit model is fitting the wrong star's geometry)
- Signals that barely pass the detection threshold (noisy fits)

ANOVA F-score: **505** — surprisingly high. This is why inclination is in the top 5 features despite seemingly being constrained to ≈90° for all transiting systems.

---

## `koi_teq` — Equilibrium Temperature (Kelvin)

### What it is
The temperature the planet would reach if it were a perfect blackbody — a theoretical maximum temperature assuming zero albedo and instant heat redistribution:

```
Teq = T★ × (Rs / 2a)^0.5 × (1-A)^0.25

Where:
T★ = stellar effective temperature
Rs = stellar radius
a  = semi-major axis
A  = Bond albedo (fraction of light reflected, typically ~0.3 assumed)
```

### Reference temperatures
```
Planet          Teq (K)
────────────────────────
Mercury         ~440
Venus           ~230 (actual surface: 735 K due to greenhouse)
Earth           ~255 (actual surface: 288 K)
Mars            ~210
Hot Jupiter     1,000–3,000
Ultra-hot Jup   > 3,000 (iron can evaporate)
Habitable zone  ~200–320
```

### Why it discriminates (observational bias, not physical)
Confirmed planets in my dataset are preferentially hot (Teq > 500 K) because close-in planets have more transits, higher SNR, and are easier to confirm. Candidates are preferentially cool (Teq ~ 200–300 K) because habitable-zone planets have few transits and are hard to follow up.

ANOVA F-score: **286** — the model is learning an observational bias, not a physical one. Close-in planets were just easier to confirm.

---

## `koi_insol` — Insolation Flux (Earth Flux Units)

### What it is
How much stellar radiation the planet receives compared to Earth:

```
Insol = (L★/L☉) × (1 AU / a)²

Earth = 1.0 (by definition)
Mars  = 0.43
Venus = 1.91
Jupiter = 0.037
Hot Jupiter at 0.05 AU ≈ 400
```

### Habitable zone range
Roughly 0.25–1.7 Earth flux units. Planets in this range receive enough energy for liquid water to potentially exist on their surfaces.

### Relationship to Teq
Insolation and equilibrium temperature are closely correlated — both increase as planets get closer to their stars. They encode similar physical information but from slightly different angles (flux is incident energy, Teq is the resulting equilibrium temperature accounting for stellar luminosity).

---

## `koi_dor` — Orbital Distance to Stellar Radius Ratio (a/Rs)

### What it is
Semi-major axis divided by stellar radius — a dimensionless geometric quantity.

```
koi_dor = a / Rs = koi_sma (AU) / (koi_srad × R☉_in_AU)
```

### Why it is in the model
This ratio appears directly in the transit duration equation and the stellar density formula. It is more geometrically fundamental than `koi_sma` alone because it normalises out the stellar size.

A small `a/Rs` means the planet is very close to the stellar surface — which is geometrically suspicious for short-period candidates that barely fit.

---

## Why All 13 Transit Features Are Kept

You might wonder: if `koi_depth` and `koi_ror` are highly correlated (r ≈ 0.90), why keep both?

Three reasons:

1. **They are not perfectly correlated** — small differences in how each is computed mean each carries slightly unique information
2. **Random Forest handles correlations gracefully** — it splits votes between correlated features but does not break
3. **The Genesis CNN's dual-branch architecture** was specifically designed for correlated features — one branch handles local correlations (adjacent correlated features), the other handles broader patterns. Removing features would undermine this design.

---

*Previous: [02 — The Three Classes](02_understanding_the_three_classes.md)*  
*Next: [04 — Stellar Parameter Features](04_feature_physics_stellar.md)*
