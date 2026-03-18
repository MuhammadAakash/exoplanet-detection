# 11 — Glossary: Astronomy and ML Terms Explained Simply

> Every term that came up during this project, explained in plain language. Written for someone with a software/data science background who is new to astronomy.

---

## Astronomy Terms

---

### Transit
When a planet passes in front of its star from our perspective, it blocks a tiny fraction of the star's light. The star dims very slightly for a short period. This dimming event is called a transit.

Think of it like a fly passing between you and a lamp — you cannot see the fly, but you can see the lamp dim momentarily.

---

### Light Curve
The record of a star's brightness over time, plotted as flux (brightness) vs time. Looks like a flat line with small dips where transits occur.

The shape of the dip — its depth, duration, and symmetry — contains all the physical information about the transiting object.

---

### Transit Depth
How much the star dims during a transit, measured in parts per million (ppm):
```
Depth = (Planet radius / Star radius)² × 10⁶
```
Small depth → small planet. Large depth → large planet (or an eclipsing binary star).

---

### ppm — Parts Per Million
A unit for measuring tiny fractional changes.
- 1 ppm = 0.0001% change
- 84 ppm = Earth transiting the Sun (0.0084% dimming)
- 10,000 ppm = Jupiter transiting the Sun (1% dimming)
- 500,000 ppm = one star eclipsing another (50% dimming)

---

### Flux
The amount of light (energy per unit area per unit time) received from a star. When a planet transits, flux decreases. The normalised flux relative to the star's baseline brightness is what light curves show.

---

### BLS — Box Least Squares Periodogram
The algorithm Kepler's pipeline used to find periodic transit signals in light curves. Tries every possible period and looks for a repeating box-shaped dip. When the correct period is tried, all transit dips stack up coherently — producing the strongest possible signal.

Like turning a radio dial through every frequency until you find the one where a song comes in clearly.

---

### TCE — Threshold Crossing Event
A periodic signal in a Kepler star's light curve that passed the automated detection threshold (MES > 7.1σ). Every KOI started as a TCE. Not every TCE became a KOI — some were filtered immediately.

---

### KOI — Kepler Object of Interest
A TCE that passed preliminary automated vetting and was flagged for further analysis. My dataset contains 3,901 KOIs, each corresponding to one potential planet signal.

---

### MES — Multiple Event Statistic
The combined signal-to-noise ratio of all observed transits stacked together:
```
MES = (transit depth × √N_transits) / noise_per_transit
```
Kepler's detection threshold was MES > 7.1 — the signal must stand more than 7.1 standard deviations above the noise.

High MES means the signal was clearly detected. It does NOT mean the signal is a planet — eclipsing binaries also have very high MES.

---

### Eclipsing Binary (EB)
Two stars orbiting each other. When one passes in front of the other from our perspective, the combined brightness dims. This looks like a planet transit but is usually:
- Much deeper (stars are much bigger than planets)
- Has a second dip at half the orbital period (when the other star passes behind)
- Often V-shaped rather than flat-bottomed (if grazing)

The most common source of false positives in Kepler data.

---

### Background Eclipsing Binary (BEB)
A faint eclipsing binary star that happens to be in the same Kepler pixel as the target star. Its brightness variations get diluted by the bright target, creating a shallow apparent transit on the target. The centroid (centre of light) shifts during the dip — the key diagnostic.

The most common source of false positives for faint Kepler targets.

---

### Centroid
The flux-weighted centre of a stellar image on the detector. For a point source (star), this is simply the position of the brightest pixel. During a genuine planet transit, the centroid should remain perfectly stationary. If it shifts, the dip is coming from a different source (background EB).

---

### Centroid Offset
When the centroid shifts during a transit dip. Indicates the signal is coming from a nearby contaminating star, not the target. Measured to milliarcsecond precision by Kepler's centroid analysis. Triggers `koi_fpflag_co`.

---

### Secondary Eclipse / Occultation
When the orbiting companion passes BEHIND the primary star (the opposite of a transit). For an eclipsing binary, the secondary star is lost behind the primary — causing a real brightness dip at orbital phase 0.5. For a planet, this event produces negligible dimming (the planet is invisible). Detecting a secondary eclipse is near-definitive proof of an eclipsing binary.

---

### Impact Parameter (b)
The sky-projected distance between the centre of the star and the transit chord, normalised by the stellar radius.
- b = 0: Planet crosses dead centre of star (deepest, flattest U-shape)
- b = 0.9: Planet grazes the stellar limb (V-shaped transit, short duration)
- b > 1.0: No transit occurs

---

### Semi-Major Axis (a)
The average orbital distance between planet and star. 1 AU = Earth-Sun distance = 149.6 million km. Derived from orbital period using Kepler's Third Law.

---

### Kepler's Third Law
The mathematical relationship between orbital period and orbital distance:
```
T² ∝ a³  (simplified)
```
More precisely: T² = (4π²/GM★) × a³

Knowing the period and stellar mass exactly determines the orbital distance. This is why `koi_period` and `koi_sma` are highly correlated.

---

### Equilibrium Temperature (Teq)
The temperature a planet would reach if it were a perfect blackbody — reflecting no light and re-radiating all absorbed stellar energy uniformly. A theoretical reference temperature, not the actual surface temperature (which depends on greenhouse effects, atmospheric circulation, etc.).

---

### Insolation Flux
How much stellar radiation the planet receives compared to Earth. Earth = 1.0. Mars = 0.43. A hot Jupiter at 0.05 AU receives ~400× Earth's flux. The habitable zone (where liquid water could exist) is roughly 0.25–1.7 Earth flux units.

---

### Stellar Photosphere
The visible "surface" layer of a star from which light escapes. The effective temperature (Teff) is the temperature of this layer. Not the core temperature (millions of Kelvin) but the surface we can see and measure (3,000–50,000 Kelvin depending on stellar type).

---

### Spectral Type
A classification system for stars based on their surface temperature: O, B, A, F, G, K, M (hottest to coolest). Kepler targeted mostly FGK stars — solar-type stars. The Sun is type G (5,778 K).

---

### Metallicity [Fe/H]
Abundance of elements heavier than hydrogen and helium in a star, relative to Solar:
- [Fe/H] = 0: Same iron abundance as the Sun
- [Fe/H] = +0.3: Twice the Solar iron abundance
- [Fe/H] = -0.3: Half the Solar iron abundance

Metal-rich stars are more likely to host giant planets (planet-metallicity correlation).

---

### log g — Stellar Surface Gravity
Surface gravity of a star, expressed as log₁₀(g in cm/s²).
- Main sequence (Sun-like) stars: log g ≈ 4.0–4.5
- Subgiants: log g ≈ 3.5–4.0
- Giants: log g ≈ 2.0–3.5

Important because giant stars misclassified as dwarfs in the Kepler Input Catalogue were a major source of false positives.

---

### Spectroscopy
Analysing the spectrum of a star's light — how the brightness varies with wavelength. Spectral lines (dark absorption lines at specific wavelengths) reveal the star's temperature, surface gravity, metallicity, and Doppler velocity. More accurate than photometry for stellar parameters but requires more telescope time.

---

### Radial Velocity
The component of a star's velocity along our line of sight, measured via the Doppler shift of spectral lines. A planet's gravity tugs its host star as it orbits — the star wobbles back and forth at the same period as the planet. Measuring this wobble gives the planet's minimum mass. One of the main confirmation methods for Kepler planets.

---

### Transit Timing Variation (TTV)
When multiple planets orbit the same star, they gravitationally perturb each other's orbits, causing transits to arrive slightly earlier or later than predicted. The pattern of timing variations uniquely identifies multi-planet systems. Cannot be produced by false positive mechanisms.

---

### Rho-Star Test (Stellar Density Test)
Comparing the stellar density derived from the transit shape (using transit timing) with the stellar density measured independently from spectroscopy. If they disagree significantly, the transit is coming from a different star (background EB). One of the most rigorous false positive diagnostics.

---

### Phase Folding
Stacking all observed transits on top of each other by mapping time to orbital phase (0 to 1). All transits align at phase 0.0. Secondary eclipses (for EBs) appear at phase 0.5. Stacking improves signal-to-noise proportionally to √N_transits.

---

### Robovetter
NASA's automated vetting algorithm applied to all Kepler TCEs after Q1-Q17 DR25 processing. Applies dozens of diagnostic tests and computes a confidence score (koi_score: 0.0–1.0) for each KOI. Also sets the false positive flags (koi_fpflag_*) used in my model.

---

### Kepler Input Catalogue (KIC)
The pre-launch catalogue of ~11 million stars that Kepler observed. Stellar parameters in the KIC were estimated from photometric colours alone — no spectroscopy. This led to systematic errors in stellar classification, particularly misidentifying evolved stars (subgiants, giants) as main sequence dwarfs.

---

### Habitable Zone
The range of orbital distances around a star where liquid water could exist on a rocky planet's surface — neither too hot (water evaporates) nor too cold (water freezes). For Sun-like stars, the habitable zone is roughly 0.95–1.67 AU. Earth sits comfortably within it.

---

### Fulton Gap (Radius Gap)
A deficit of planets with radii between approximately 1.5 and 2.0 Earth radii, discovered from Kepler data in 2017 (Fulton et al. 2017). This gap separates:
- Rocky super-Earths (< 1.5 R⊕): Lost their atmospheres through photoevaporation
- Gas mini-Neptunes (> 2.0 R⊕): Retained gaseous envelopes

The gap is a direct imprint of atmospheric physics and is visible in the confirmed planet population.

---

### Charge Bleeding / Column Bleeding
A CCD detector artefact where very bright stars generate excess electrons that bleed along the CCD column. If a known eclipsing binary is bright enough to bleed, its periodic signal can contaminate nearby pixels — creating false transit signals on target stars whose pixels happen to be along the bleeding path. Triggers `koi_fpflag_ec`.

---

### Dilution
When a nearby star's light contaminates the Kepler pixel containing the target, its constant light contribution reduces the apparent transit depth:
```
Depth_observed = Depth_true × (L_target / (L_target + L_contaminant))
```
A 50% eclipsing binary diluted by a factor of 100 appears as a 0.5% transit — planet-like. The magnitude difference between target and contaminating source quantifies this dilution.

---

## Machine Learning Terms (Exoplanet Context)

---

### Class Imbalance
When one class has far more training examples than another. In my dataset: 4.9× more CONFIRMED (2,341) than CANDIDATE (477). Without correction, models learn to predict the majority class.

---

### Majority Class Baseline
The accuracy achieved by a model that predicts the most common class for every input. For my dataset: 60% (always predict CONFIRMED). Any real model must beat this meaningless baseline.

---

### Class Weight Balancing
Assigning higher training weight to minority class examples so the model pays equal attention to all classes during training. Weight = 1/frequency → rare classes weighted more heavily.

---

### Stratified Split
A train/val/test split that preserves the class proportions from the original dataset in each split. Without stratification, random splits might under-represent minority classes.

---

### Data Leakage
When information from the test set (or future data) is used during training, making the model appear to perform better than it genuinely would on new data.

In this project:
- Fitting the imputer on all data (including test) → leakage
- Including koi_score (which was used to generate labels) → leakage
- Including koi_pdisposition (preliminary human label) → leakage

---

### ANOVA F-Score
Statistical test for whether a feature's distribution is significantly different across class labels. F = between-class variance / within-class variance. High F → feature is discriminative. Used in Section 7 of EDA for pre-modelling feature assessment.

---

### Median Imputation
Replacing missing values with the median of the observed values for that feature. Robust to outliers because median is unaffected by extreme values. The correct strategy when distributions are right-skewed.

---

### StandardScaler
Transforms each feature to have mean ≈ 0 and standard deviation ≈ 1 by subtracting the mean and dividing by std. Affected by outliers (outliers inflate std). Must be fitted on training data only.

---

### Macro F1
Average F1 score across all classes, treating each class equally regardless of frequency. Penalises models that ignore minority classes. Primary evaluation metric for this project due to class imbalance.

---

### Cohen's Kappa
Measures agreement between predictions and true labels, correcting for the agreement expected by chance given the class frequencies. Ranges from -1 (worse than chance) to 1 (perfect). More honest than accuracy for imbalanced datasets.

---

### Matthews Correlation Coefficient (MCC)
A single metric capturing all four cells of the confusion matrix simultaneously. Ranges from -1 to 1. Considered the most informative single metric for imbalanced multiclass classification. Particularly useful when classes are very imbalanced.

---

### Convolutional Neural Network (CNN)
A neural network that applies convolutional filters — small windows that slide across the input looking for local patterns. Originally designed for images (2D convolution). For tabular data: applied as 1D convolution, where each filter looks at a small window of adjacent features. The Genesis CNN applies two parallel branches of 1D convolution with different window sizes to capture both local and broader feature interactions.

---

### Kernel Size (CNN)
The width of the convolutional filter window — how many adjacent features each filter "sees" at once. Small kernel (3): captures tight local interactions. Large kernel (7): captures broader patterns across more features. The Genesis CNN uses both sizes in parallel branches.

---

### Early Stopping
Training the model and stopping when performance on the validation set stops improving (or gets worse). Prevents overfitting — the model is stopped before it memorises the training data. Patience parameter defines how many epochs of non-improvement to tolerate before stopping.

---

### Macro vs Weighted Metrics
- **Macro:** Compute metric separately for each class, take simple average. Each class contributes equally.
- **Weighted:** Compute metric separately for each class, take weighted average (by class frequency). Common classes contribute more.

For imbalanced datasets, macro metrics are preferred because they treat all classes equally regardless of frequency.

---

*This glossary will grow as the project progresses into modelling stages.*

---

*Previous: [10 — Preprocessing Decisions](10_preprocessing_decisions.md)*  
*Return to: [README — Notes Index](README.md)*
