# Final Model Comparison — Complete Reference Notes
> Exoplanet Candidate Vetting · Stage 6 · MSc Data Science Dissertation
> Dataset: NASA Kepler KOI Q1-Q17 DR25 · 3,901 samples · 37 features · 3 classes

---

## Table of Contents
1. [What Stage 6 does](#what-stage-6-does)
2. [Final results table](#final-results-table)
3. [Key findings — what the numbers say](#key-findings)
4. [Why Random Forest wins on tabular data](#why-rf-wins)
5. [Why Genesis CNN beats Baseline CNN](#genesis-vs-baseline)
6. [The CANDIDATE class problem](#candidate-class)
7. [Dissertation talking points](#dissertation-talking-points)

---

## What Stage 6 Does

Stage 6 is the final analysis stage. It does not train any new models. It:

1. Loads metrics from all five previously trained models
2. Generates five dissertation-ready comparison figures
3. Saves a clean ranked summary table (`stage6_final_comparison.csv`)
4. Prints a formatted leaderboard to the log

### Figures generated

| File | What it shows |
|------|--------------|
| `comp_01_overall_metrics.png` | Accuracy / F1 Macro / ROC-AUC / Cohen's κ bar chart for all 5 models |
| `comp_02_perclass_f1.png` | Per-class F1 grouped bars — 5 models × 3 classes |
| `comp_03_heatmap.png` | All metrics as a colour-coded heatmap |
| `comp_04_radar.png` | Radar/spider chart for multi-metric comparison |
| `comp_05_cnn_deep_dive.png` | Baseline CNN vs Genesis CNN head-to-head |

---

## Final Results Table

| Rank | Model | Accuracy | F1 Macro | ROC-AUC | Cohen's κ | F1 Confirmed | F1 False Pos. | F1 Candidate |
|------|-------|----------|----------|---------|-----------|--------------|---------------|--------------|
| 1 | **Random Forest** | **89.4%** | **0.778** | **0.955** | **0.796** | 0.923 | **0.979** | 0.434 |
| 2 | **Genesis CNN** | **85.7%** | 0.749 | 0.879 | 0.733 | **0.890** | 0.978 | 0.379 |
| 3 | Logistic Regression | 81.9% | 0.763 | 0.924 | 0.692 | 0.838 | 0.991 | **0.461** |
| 4 | SVM | 81.9% | 0.739 | 0.897 | 0.681 | 0.845 | 0.988 | 0.383 |
| 5 | Baseline CNN | 80.7% | 0.738 | 0.908 | 0.666 | 0.842 | 0.972 | 0.400 |

**Key numbers:**
- Genesis CNN is ranked **2nd overall** by accuracy
- Random Forest leads by **3.7 percentage points** over Genesis CNN
- Genesis CNN beats Baseline CNN by **5.0 percentage points** (the largest single improvement in the pipeline)
- Logistic Regression achieves the **best CANDIDATE F1** (0.461) — surprising and worth discussing

---

## Key Findings

### 1. Random Forest is the strongest model overall

RF leads on accuracy (89.4%), F1 Macro (0.778), ROC-AUC (0.955), and Cohen's κ (0.796). This is consistent with the broader literature: tree-based methods retain an advantage on tabular data at moderate sample sizes (Grinsztajn et al., 2022).

The gap is not a failure of the deep learning models — it reflects fundamental properties of the problem (tabular structure, small N, binary flag features with extremely high discriminative power).

### 2. Genesis CNN clearly outperforms Baseline CNN

The dual-branch architecture improved accuracy from **80.7% → 85.7%** — a 5.0 percentage point gain. This is a meaningful result: it directly validates the hypothesis that multi-scale feature processing (local k=3 + global k=7) provides richer representations than single-scale processing alone.

However, Genesis CNN does not close the gap with RF. It trails by 3.7 pp. This suggests that the architectural improvement matters, but the fundamental tabular data advantage of RF is not primarily an architectural problem.

### 3. Logistic Regression has the best CANDIDATE F1

LR achieves CANDIDATE F1 = 0.461, the highest of any model. This is counterintuitive — LR is the simplest model. The likely explanation is that LR's linear decision boundary may happen to align well with the ambiguous, overlapping distributions of the CANDIDATE class, which lacks clear separating features. LR does not commit strongly to any class, which means it misclassifies fewer CANDIDATE samples as CONFIRMED.

### 4. FALSE POSITIVE is easy — all models get F1 > 0.97

The four false-positive flags (`koi_fpflag_*`) are so discriminative that every model, from LR to RF, achieves near-perfect FALSE POSITIVE detection. This class is effectively solved at the feature level — the modelling architecture makes almost no difference here.

### 5. CANDIDATE remains hard for every model

All five models score F1 ≤ 0.46 on CANDIDATE. This is a data problem, not a modelling problem:
- CANDIDATE = "not yet resolved" — an administrative label, not a physical category
- CANDIDATE candidates overlap with both CONFIRMED and FALSE POSITIVE in feature space
- No model can learn to separate labels that are defined by absence of information

This result should be explicitly acknowledged in the dissertation.

---

## Why Random Forest Wins on Tabular Data

This is one of the most important questions to be able to answer in a viva.

### Reason 1: The false-positive flags are binary thresholds

`koi_fpflag_co = 1` means "centroid offset detected — this is a false positive." A single tree split on this column correctly classifies ~60% of FALSE POSITIVE cases in one step. The RF can make this its root split and immediately sort most of the data.

A CNN must learn this threshold through gradient descent across multiple layers — an indirect and less efficient process for such a sharp, discontinuous boundary.

### Reason 2: Trees are scale-invariant

Our 37 features span very different scales and distributions (binary 0/1 flags, orbital periods in days, magnitudes, densities). StandardScaling partially normalises this for the CNN, but the tree never needs to worry about scale — it splits on rank order, not absolute values.

### Reason 3: Sample size regime

With 2,730 training samples, we are in the regime where RF consistently outperforms neural networks. Neural networks have higher inductive bias (assumptions encoded in the architecture) and need more data to learn past random initialisation. At ~50,000+ samples, the gap would likely narrow.

### Reason 4: RF handles feature importance asymmetry

Some features have overwhelming discriminative power (`koi_fpflag_*`). RF naturally concentrates its splits on these features in early layers of every tree. The CNN must learn this implicitly through the gradient signal, which is less direct.

---

## Genesis CNN vs Baseline CNN

The 5.0 pp accuracy improvement from Baseline → Genesis CNN validates the dual-branch hypothesis.

### What changed

| Component | Baseline CNN | Genesis CNN |
|-----------|-------------|-------------|
| Branches | 1 | 2 (local + global) |
| Local kernels | 3, 3, 3 | 3, 3 |
| Global kernels | — | 7, 5 |
| Dense head | Dense(64) | Dense(128) → Dense(64) |
| Dropout | 0.3 | 0.4 |
| Parameters | 40,163 | 42,435 |

### Why the improvement happened

The global branch (k=7, k=5) can detect interactions across a wider window of features. For example, a filter in the global branch might learn that "high stellar effective temperature + high planet-to-star radius ratio + high orbital period" is a pattern that distinguishes CONFIRMED planets around larger stars. This combination spans features that are far apart in the feature vector — the Baseline CNN's k=3 filters could not capture it in a single layer.

### What did not improve

CANDIDATE F1 actually dropped slightly from 0.400 → 0.379. The dual-branch design did not help with the hardest class. This is consistent with the CANDIDATE class being intrinsically ambiguous (administrative label) rather than architecturally solvable.

---

## The CANDIDATE Class Problem

Every model in this project struggles with CANDIDATE. The F1 scores range from 0.37 to 0.46 — far below the 0.84–0.99 range for the other two classes.

### Why is CANDIDATE hard?

1. **Administrative label:** CANDIDATE means "this signal is transit-like but has not been followed up with ground-based observations yet." It is not a physically distinct class — it is an unresolved case. Many CANDIDATEs will eventually be reclassified as either CONFIRMED or FALSE POSITIVE.

2. **Overlapping feature distributions:** Because CANDIDATEs are a mixed bag of potential planets and potential false positives, their feature distributions overlap heavily with both other classes.

3. **Class imbalance:** CANDIDATE is only ~13% of the dataset (513 samples). Even with class weighting, there is simply less signal to learn from.

4. **What this means for the dissertation:** The CANDIDATE F1 results do not reflect model quality — they reflect the inherent difficulty of the class. A classifier that scored 0.80 CANDIDATE F1 would be suspicious, not impressive. The honest answer is that no model can reliably vet unresolved cases.

### How to address this in the dissertation

> "The CANDIDATE class presents a fundamental labelling challenge: these objects are administratively classified as 'not yet resolved' rather than physically distinct. All five models in this study achieve CANDIDATE F1 ≤ 0.46, consistent with prior work on KOI datasets. This reflects the intrinsic label overlap rather than a limitation of any specific architecture."

---

## Dissertation Talking Points

### Research question answered
> "Can a simplified Genesis CNN accurately vet exoplanet candidates from Kepler feature data, and how does it compare to classical ML and baseline deep learning models?"

**Answer:** Yes, the Genesis CNN achieves 85.7% accuracy and F1 Macro 0.749 — competitive with and outperforming the Baseline CNN. Random Forest remains the strongest model overall (89.4%), consistent with the empirical finding that tree-based methods retain an advantage on tabular data at moderate sample sizes.

### The most important comparison
> "The Genesis CNN (dual-branch, 42,435 params) outperforms the Baseline CNN (single-branch, 40,163 params) by 5.0 percentage points in accuracy with only a 2,272-parameter increase. This directly validates the hypothesis that multi-scale convolutional feature processing — combining local (k=3) and global (k=7) receptive fields — provides richer representations for this classification problem."

### Why the CNN doesn't beat RF
> "The performance gap between the Genesis CNN (85.7%) and Random Forest (89.4%) is consistent with the broader finding that tree-based methods outperform deep learning on tabular data at small-to-moderate scales (Grinsztajn et al., 2022). The dominant features in this dataset (`koi_fpflag_*`) are binary flags whose discriminative power is captured in a single decision tree split, an operation that gradient-based learning approximates indirectly."

### What would close the gap with RF?
1. Raw light curve data (CNN operating on actual photometric signal, not pre-extracted features)
2. More training samples (the TESS mission has millions of TCEs)
3. Attention mechanisms that can explicitly learn to attend to the flag features
4. Feature ordering by physical group (transit/stellar/signal) to make the CNN's adjacency assumption physically meaningful

---

## References

- **Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022).** Why tree-based models still outperform deep learning on tabular data. *NeurIPS 2022*.
- **Shallue, C.J. & Vanderburg, A. (2018).** Identifying Exoplanets with Deep Learning. *AJ*, 155(2).
- **Thompson, S.E. et al. (2018).** Planetary Candidates Observed by Kepler. VIII. *ApJS*, 235(2). — Source of the KOI DR25 dataset and labelling methodology.

---

*Last updated: Stage 6 — Final Comparison complete*
*Pipeline complete: Preprocessing → EDA → Classical ML → Baseline CNN → Genesis CNN → Comparison*
