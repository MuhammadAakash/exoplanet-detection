# 01 — How Kepler Finds Planets (The Physics)

> This is the foundation of the entire project. If you understand this page, everything else — the features, the EDA, the model — makes sense.

---

## The Core Idea in One Sentence

When a planet passes in front of its star, it blocks a tiny fraction of the star's light, and Kepler measured that tiny dimming from space.

---

## The Transit Method — Step by Step

Imagine you are standing far away from a lamppost at night. A moth flies between you and the lamp. For a moment, the lamp dims slightly. You cannot see the moth — it is too small and far away — but you can see the lamp get dimmer. That is exactly what Kepler detected. Stars are the lampposts. Planets are the moths.

```
What Kepler sees from space:

Normal brightness:
★ ════════════════════════════════ (steady light curve)

Planet approaching:
★ ══════╗                          (brightness starts dropping — ingress)
        ║
        ╚══════════════════════════ (planet fully in front — flat bottom)

Planet leaving:
★                          ╔══════ (brightness rising — egress)
                           ║
═══════════════════════════╝

Time →  |--ingress--|--flat bottom--|--egress--|
```

This dip in brightness is called a **transit**. The shape of the dip is called the **light curve**.

---

## The Key Equation — How Big Is the Planet?

The depth of the dip tells you the size of the planet relative to the star:

```
Transit Depth = (Planet Radius / Star Radius)²

Or:  depth = (Rp / Rs)²
```

**Real examples:**
| Object | Size Ratio | Depth |
|--------|-----------|-------|
| Earth passing in front of Sun | 1/109 | 0.0084% = 84 ppm |
| Jupiter passing in front of Sun | 1/10 | ~1% = 10,000 ppm |
| Star passing in front of star (eclipsing binary) | ~1/1 | up to 50% = 500,000 ppm |

**ppm = parts per million.** A depth of 84 ppm means the star gets 0.0084% dimmer. Kepler could detect this from 3,000 light years away.

---

## How Kepler Actually Worked

### The Telescope

Kepler was launched in 2009 and stared at the same patch of sky for 4 years without blinking. It monitored **150,000 stars simultaneously**, measuring each star's brightness every 30 minutes.

It had to be in space because Earth's atmosphere blurs starlight too much. Ground-based telescopes can detect brightness changes of ~1,000 ppm. Kepler achieved **20–30 ppm** — about 50× better.

### Finding the Transits — Box Least Squares (BLS)

After 4 years of data, each star had ~70,000 brightness measurements. NASA's pipeline then searched for repeating dips using an algorithm called **Box Least Squares (BLS)**:

1. Try every possible period from 0.5 days to 500 days
2. At each trial period, fold the light curve (stack all measurements on top of each other)
3. Look for a consistent box-shaped dip at one particular orbital phase
4. The period that produces the clearest, deepest box dip is the winner

```
Phase folding at the correct period:

Transit 1:          Transit 2:          Transit 3:
    ╔══╗                ╔══╗                ╔══╗
════╝  ╚════        ════╝  ╚════        ════╝  ╚════

All three stacked on top of each other:
    ╔══╗
════╝  ╚════   ← Clean, deep signal — this is the correct period
```

### The Detection Threshold — MES > 7.1

NASA required the stacked signal to be at least **7.1 sigma** above the noise level before flagging it as a candidate. This is the **Multiple Event Statistic (MES)**. The threshold of 7.1 was chosen to limit false alarms to about 1 per year across the entire mission.

Any signal passing this threshold became a **Threshold Crossing Event (TCE)**. TCEs that passed further automated vetting became **Kepler Objects of Interest (KOIs)** — which is exactly what my dataset contains.

---

## Why Most Signals Are NOT Planets

Of 3,901 KOIs in my dataset, only 2,341 (60%) are confirmed planets. The rest are false positives or candidates. Why?

Because anything that causes a **periodic dimming** will be flagged. And many things cause periodic dimming that have nothing to do with planets:

**Two stars orbiting each other (eclipsing binary):**  
When one star passes in front of the other, you get a dip. It looks almost identical to a planet transit — except the dip is much deeper (stars are much bigger than planets) and there is a second dip when the other star passes behind the first.

**A faint background star that is itself an eclipsing binary:**  
If a faint eclipsing binary happens to be in the same pixel as your target star, its variability gets mixed into the target's light curve. The dip appears shallow (because the bright target dilutes it) and looks planet-like.

**Stellar variability:**  
Stars have spots, pulsations, and flares. Some of these can mimic transits if they are regular enough.

**Instrumental noise:**  
Cosmic rays, detector systematics, and telescope pointing jitter can all create spurious signals.

**Teaching the model to tell the difference between these scenarios is exactly what my dissertation is about.**

---

## Why Kepler's Photometric Precision Matters

Kepler achieved 20–30 ppm precision on bright stars. To put this in perspective:

- That is like measuring the change in brightness of a car headlight when a mosquito flies past it — from 10 miles away
- Earth-sized planets produce dips of ~84 ppm — only 3–4× the noise floor
- Finding Earth-like planets genuinely required the best space telescope ever built for this purpose

This also means that many signals in the dataset are barely above the noise threshold. These marginal signals are disproportionately represented in the **CANDIDATE** class — they passed the 7.1σ threshold but there is not enough margin to confidently confirm them.

---

## The Key Physical Insight That Shapes Everything

Planets are small → small dips → small depth  
Stars are large → large dips → large depth  

This one physical fact means that:
- **Transit depth** is one of the most informative features in my dataset
- Anything with depth > ~50,000 ppm is almost certainly not a planet
- The false positive flags are essentially different ways of testing whether the dip is planetary-scale or stellar-scale

---

## What This Means for My Model

Every feature in my dataset is either:
1. **Measuring the transit shape** — to check if it looks like a planet dip or something else
2. **Measuring the host star** — to correctly interpret what the transit depth means in physical units
3. **Running a diagnostic test** — directly checking for known false positive signatures

Understanding this structure is why my EDA makes sense. I am not just exploring random numbers — I am exploring physical measurements that each tell a specific story about whether this signal is a planet.

---

*Next: [02 — Understanding the Three Classes](02_understanding_the_three_classes.md)*
