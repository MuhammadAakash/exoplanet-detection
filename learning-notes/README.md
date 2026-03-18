# 📚 Exoplanet Project — Learning Notes

> Personal notes written in plain human language during my MSc Data Science dissertation.  
> Topic: **Exoplanet Candidate Vetting Using a Simplified Genesis CNN**  
> Dataset: NASA Kepler Objects of Interest (KOI) Q1-Q17 DR25

---

## 📂 Files in This Folder

| File | What It Covers |
|------|---------------|
| `01_how_kepler_finds_planets.md` | The physics of transit detection — how Kepler actually works |
| `02_understanding_the_three_classes.md` | CONFIRMED vs FALSE POSITIVE vs CANDIDATE — who they are and why |
| `03_feature_physics_transit.md` | Every transit geometry feature explained in plain English |
| `04_feature_physics_stellar.md` | Every stellar parameter feature explained in plain English |
| `05_feature_physics_fp_flags.md` | The four false positive flags — the most powerful features |
| `06_feature_physics_signal_and_magnitudes.md` | Signal quality and photometric magnitude features |
| `07_eda_what_i_explored_and_why.md` | All 8 EDA sections — the logic behind each one |
| `08_eda_key_findings_and_decisions.md` | What the EDA found and how each finding shaped the model |
| `09_feature_engineering_decisions.md` | Why I added 3 new features and what I deliberately did NOT do |
| `10_preprocessing_decisions.md` | Every preprocessing choice and the reasoning behind it |
| `11_glossary.md` | Astronomy and ML terms explained simply |

---

## 🧠 How to Read These Notes

These are **not** textbook notes. They are written the way I actually understood each concept — by connecting the physics to the data to the modelling decision.

Every note answers three questions:
1. **What is this?** — plain English definition
2. **Why does it matter?** — the physical or statistical reason
3. **What does it mean for the model?** — the downstream modelling decision

---

## 🗂️ Project Context

```
Research Question:
Can a simplified Genesis CNN accurately vet exoplanet candidates
from Kepler light curve data, and how does it compare to classical
ML and baseline deep learning models?

Dataset: 3,901 KOIs | 37 model features | 3 classes
Pipeline: Preprocessing → EDA → Classical ML → Baseline CNN → Genesis CNN → Comparison
```

---

*Written during MSc Data Science dissertation — University project*  
*Last updated: Stage 2 (EDA) complete*
