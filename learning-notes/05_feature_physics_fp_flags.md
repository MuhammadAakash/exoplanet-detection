# 05 — Feature Physics: False Positive Flag Features

> These four binary features are the single most powerful group in the entire dataset. Three of them are in the top four ANOVA scores. Understanding WHY they are powerful tells you almost everything about the classification problem.

---

## The Four Flags at a Glance

| Feature | ANOVA F-Score | Rank | What It Catches |
|---------|--------------|------|----------------|
| `koi_fpflag_co` | **2,197** | #1 | Background eclipsing binaries (centroid shift) |
| `koi_fpflag_ss` | **1,357** | #2 | All eclipsing binaries (secondary eclipse) |
| `koi_fpflag_ec` | **695** | #3 | Known contaminating eclipsing binaries |
| `koi_fpflag_nt` | **~400** | ~#5 | Grazing systems and non-planetary shapes |

All four are **binary indicators: 0 = test passed (no flag), 1 = test failed (flag raised).**

The gap between these F-scores and the next feature (koi_incl at F=505) shows how dominant these flags are. They are essentially automated versions of the tests an astronomer would apply.

---

## Why Four Separate Flags?

Because the four most common false positive mechanisms each leave a **different physical fingerprint** in the data. Each flag tests for a specific fingerprint. Together they cover most known false positive scenarios.

Think of them as four independent detectors, each tuned to one type of impostor.

---

## `koi_fpflag_co` — Centroid Offset Flag

**ANOVA F-Score: 2,197 — Highest in the entire dataset**

### What it tests
Does the **centroid** (the flux-weighted centre of the stellar image) shift position during the brightness dip?

### The physics — why this is so powerful

During a genuine planet transit, everything about the stellar image should remain perfectly stationary. The planet is invisible (too small, too faint). The star does not move. The centre of light does not move. Only the brightness changes.

If the centroid **does** shift during the dip, it means the dip is not coming from the target star. The light is being pulled toward a different, brighter source that is getting dimmer — a nearby star that is an eclipsing binary.

```
No centroid offset (genuine planet transit):
                  Target star
                     ★
The centroid        ✦ ← stays here throughout transit
                    
Centroid offset (background eclipsing binary):
                  Target star    Background EB
                     ★               ☆ (eclipsing, getting dimmer)

During the dip: the background EB gets dimmer → its contribution to the
centroid position decreases → centroid shifts TOWARD the brighter target

The centroid        ✦ → shifts this way → indicates background star
```

### What "background eclipsing binary" means
Kepler's pixels were large — about 4 arcseconds × 4 arcseconds. At typical Kepler target distances (hundreds to thousands of light years), many background stars can fit in the same pixel. If one of those background stars is an eclipsing binary (two stars orbiting each other), its brightness variations get diluted into the target star's pixel.

The diluted signal looks like a planet transit:
- It is periodic
- It is shallow (because the EB signal is diluted by the bright target)
- It has the right shape

But the centroid shifts because the source of the brightness variation is slightly off-centre.

### Sensitivity
NASA's centroid analysis can detect offsets as small as a few milliarcseconds — small enough to catch contamination from sources several arcseconds away from the target.

### When flag = 1
The centroid shifted during the transit. The dip is almost certainly coming from a nearby contaminating star, not the target. This is among the strongest possible evidence that a KOI is a false positive.

> **My note:** Learning about this feature was the moment the whole project clicked for me. A simple centroid measurement — just tracking where the light is coming from — is more powerful than all 13 transit geometry features combined. The physics of WHERE the signal comes from is more informative than the physics of WHAT the signal looks like. That is a profound insight about this classification problem.

---

## `koi_fpflag_ss` — Significant Secondary Eclipse Flag

**ANOVA F-Score: 1,357 — Second highest in dataset**

### What it tests
Is there a statistically significant brightness dip at **orbital phase 0.5** — exactly halfway through the orbital cycle?

### The physics — why phase 0.5 is special

In a **planet-star system:**
- Phase 0.0: Planet passes in front of star (primary transit — what Kepler detected)
- Phase 0.5: Planet passes BEHIND the star (secondary eclipse)
- The planet reflects/emits negligible light compared to the star
- When the planet hides behind the star: essentially nothing measurable happens
- **There is no detectable secondary eclipse for a genuine planet** (at optical wavelengths, for most planets)

In an **eclipsing binary (two stars):**
- Phase 0.0: Secondary star passes in front of primary (primary eclipse)
- Phase 0.5: Secondary star passes BEHIND primary (secondary eclipse)
- The secondary star emits real, significant light
- When the secondary hides: you lose the secondary star's light contribution
- **A real brightness dip occurs at exactly phase 0.5**

```
Planet transit signal:          Eclipsing binary signal:
Phase:  0.0        0.5          Phase:  0.0          0.5

████                             ████         ██
    ████████████████                 ████████  ██████
                                (primary)    (secondary)
    No secondary dip                 Real secondary dip
```

The depth of the secondary eclipse reveals the temperature ratio of the two stars:
```
Secondary depth / Primary depth ≈ (T₂/T₁)⁴ × (R₂/R₁)²
```

### When flag = 1
A significant dip was detected at phase 0.5. This is essentially **definitive proof of an eclipsing binary.** No planet mechanism produces a secondary eclipse at optical wavelengths.

The one exception: very hot Jupiters (Teq > 2,500 K) emit enough thermal radiation that their occultation produces a tiny dip (~50–100 ppm). The Kepler Robovetter accounts for this — the threshold for setting the flag is calibrated to avoid flagging genuine hot Jupiter secondaries.

> **My note:** This feature was the most elegant thing I learned in this project. The secondary eclipse is nature's own fingerprint for binary stars — it physically cannot occur for a planet. It is not a statistical test — it is a direct physical consequence. When this flag is set, there is almost no ambiguity. That is why it scores F=1,357 — even higher than most diagnostic tests astronomers can devise.

---

## `koi_fpflag_ec` — Ephemeris Match Flag

**ANOVA F-Score: 695 — Third highest in dataset**

### What it tests
Does the **period and transit timing** of this KOI exactly match a **known eclipsing binary** catalogued elsewhere in the Kepler field?

### The physics — charge bleeding and contamination

Kepler's CCD detectors suffered from a phenomenon called **charge bleeding** (also called column bleeding):
- When a very bright star falls on a CCD pixel, it generates enormous numbers of electrons
- The pixel "fills up" and excess electrons bleed along the CCD column
- The bleeding pattern is fixed in detector coordinates (depends on where on the chip the star falls)

If a known bright eclipsing binary (from the Kepler Eclipsing Binary Catalogue) is bright enough to bleed, its periodic signal can contaminate other stars whose Kepler pixels happen to sit along the bleeding column.

### The catalogue matching
NASA maintained a comprehensive Kepler Eclipsing Binary Catalogue (Prša et al., Kirk et al.). For each KOI, the pipeline checks:
- Does any known EB in the catalogue have the same period (within measurement error)?
- Does the transit timing match?
- Is the known EB in a position on the CCD where contamination of this target is geometrically possible?

If all three conditions are met, the `koi_fpflag_ec` flag is set to 1.

### When flag = 1
This KOI's signal is almost certainly contamination from a known eclipsing binary via charge bleeding or pixel contamination. The period match is a mathematical fingerprint that is essentially impossible to fake.

> **My note:** I initially did not understand why "checking against a catalogue" would be a model feature. Then I realised: the flag encodes real physical information — that this specific period appears in a known contaminating source. The model does not need to know the catalogue exists; it just needs to learn that when this flag is 1, the KOI is almost always a false positive. The physics (charge bleeding) is why the catalogue match means what it means.

---

## `koi_fpflag_nt` — Not Transit-Like Flag

**ANOVA F-Score: ~400 — Top 5 in dataset**

### What it tests
Does the shape of the brightness dip match the expected shape of a genuine planetary transit?

A planetary transit has a specific, predictable shape:
- Smooth ingress (planet's leading edge crossing the stellar limb)
- Flat bottom (planet fully in front, blocking same amount of light)
- Smooth egress (planet's trailing edge leaving stellar limb)
- Equal ingress and egress durations (symmetric)
- Depth is consistent from transit to transit

Any significant deviation from this expected shape raises the flag.

### What triggers the flag (set = 1)

**V-shaped transits — grazing eclipsing binaries:**
```
Expected planet shape (U-shaped):    EB grazing shape (V-shaped):
    ╔════╗                                  ╱╲
════╝    ╚════                         ═════╱  ╲═════
(flat bottom — solid disc transit)   (no flat bottom — grazing stars)
```
When one star grazes the edge of the other, the eclipsed area continuously changes throughout the eclipse — producing a V-shape with no flat bottom.

**Asymmetric transits:**
Ingress much longer than egress, or vice versa. A genuine planetary transit is symmetric because the planet is a solid disc entering and leaving at the same speed.

**Variable depth:**
If the dip is deeper on some transits and shallower on others at the same period, something is modulating the signal — stellar activity, contamination, or EB ellipsoidal variation.

**Negative transits:**
A flux increase rather than decrease. Instruments artifacts, stellar flares, or optical effects can produce upward spikes that look periodic.

**Sinusoidal shape:**
Rather than a sharp dip, a smooth sinusoidal modulation at the orbital period. This is often caused by stellar ellipsoidal deformation — the gravity of a massive companion (another star, not a planet) distorts the star into an egg shape, modulating brightness at half the orbital period.

### When flag = 1
The transit shape is inconsistent with a solid disc (planet) crossing a stellar disc. Could be a grazing binary, stellar variability, or an instrumental artefact.

---

## The Four Flags Together — The Cumulative Suspicion

The flags are designed to catch different impostor types. But they are also **complementary** — a genuine eclipsing binary should trigger multiple flags simultaneously:

| False Positive Type | `koi_fpflag_nt` | `koi_fpflag_ss` | `koi_fpflag_co` | `koi_fpflag_ec` |
|--------------------|:---:|:---:|:---:|:---:|
| Grazing EB | ✓ (V-shape) | ✓ (secondary) | maybe | maybe |
| Background EB | maybe | ✓ (secondary) | ✓ (centroid) | maybe |
| Known contaminating EB | maybe | maybe | maybe | ✓ (match) |
| Stellar variability | ✓ (shape) | | | |

For confirmed planets: all four should be 0.

This co-occurrence pattern appeared clearly in the EDA Section 5 (astrophysical relationships) — the flag co-occurrence heatmap showed that false positives have multiple flags firing simultaneously in physically consistent patterns.

---

## Why These Four Features Dominate All Others

A one-sentence summary for each:

- `koi_fpflag_co`: Tells you if the signal is from the wrong star entirely
- `koi_fpflag_ss`: Tells you if the companion is stellar (emitting light) rather than planetary
- `koi_fpflag_ec`: Tells you if this exact signal is known to come from a contaminating source
- `koi_fpflag_nt`: Tells you if the signal shape is inconsistent with a planet at all

When any of these fires, there is a specific, physically sound reason to suspect the signal is not a planet. These are not statistical correlations — they are direct physical diagnostics. That is why they dominate the ANOVA and will dominate the Random Forest feature importances.

---

## The Engineered Feature: `fp_flag_sum`

Because individual flags tell you which tests failed, the sum tells you how many:

```python
df['fp_flag_sum'] = (
    df['koi_fpflag_nt'] +
    df['koi_fpflag_ss'] +
    df['koi_fpflag_co'] +
    df['koi_fpflag_ec']
)
```

A KOI with 3 flags set is qualitatively more suspicious than one with 1 flag set — multiple independent tests all pointing toward false positive. The sum (0–4) encodes this cumulative evidence in a single feature that no individual flag provides.

---

*Previous: [04 — Stellar Parameter Features](04_feature_physics_stellar.md)*  
*Next: [06 — Signal Quality and Magnitude Features](06_feature_physics_signal_and_magnitudes.md)*
