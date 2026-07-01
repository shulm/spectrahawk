# SpectraHawk Technical Note

## Leakage Studies

To quantify the impact of recording-level data leakage on RF drone
classification, we conducted three studies. They show that naive segment-level
cross-validation severely overestimates generalization when segments from the
same continuous flight recording are split across train and test.

### Study 1: Controlled Synthetic Leakage Simulation

A generative model places a class signal on a single discriminative axis
(separation Δ, true single-segment Bayes ≈ 0.65) and a per-recording nuisance on
the remaining non-discriminative axes (scale λ). Each recording has one class, so
a model that recognizes the recording can predict the class. We compare a naive
segment-level split (`StratifiedKFold`) with an honest recording-grouped split
(`StratifiedGroupKFold`), reporting balanced accuracy as **mean ± 95% CI over 10
dataset seeds**.

**Sweep 1: nuisance λ at R = 8 recordings/class** (True Bayes ≈ 0.65)

| λ | RF Naive | RF Grouped | LR Grouped |
|---|----------|------------|------------|
| 0.0 | 0.628 ± 0.013 | 0.617 ± 0.010 | 0.632 ± 0.009 |
| 0.5 | 0.735 ± 0.015 | 0.600 ± 0.018 | 0.589 ± 0.025 |
| 1.0 | 0.903 ± 0.014 | 0.566 ± 0.040 | 0.574 ± 0.047 |
| 2.0 | 0.995 ± 0.002 | 0.515 ± 0.068 | 0.584 ± 0.076 |
| 4.0 | 1.000 ± 0.000 | 0.497 ± 0.063 | 0.584 ± 0.095 |
| 8.0 | 1.000 ± 0.000 | 0.492 ± 0.071 | 0.571 ± 0.101 |

**Sweep 2: recordings R per class at λ = 2.0**

| R | RF Naive | RF Grouped |
|---|----------|------------|
| 2 | 0.999 ± 0.002 | 0.495 ± 0.134 |
| 4 | 0.998 ± 0.002 | 0.608 ± 0.083 |
| 8 | 0.995 ± 0.002 | 0.515 ± 0.068 |
| 16 | 0.985 ± 0.003 | 0.502 ± 0.048 |
| 32 | 0.972 ± 0.003 | 0.543 ± 0.030 |

**Finding.** As the nuisance λ grows, naive evaluation climbs to 1.0 (it memorizes
the recording through its high-variance nuisance signature), while honest grouped
evaluation declines to chance; the inflation gap reaches ≈ 0.5. The Random Forest's
honest accuracy approaches chance and can dip slightly below in individual
finite-recording draws (shortcut learning on the high-variance nuisance axes);
Logistic Regression declines toward chance while retaining only a weak vestige of
the true signal. Sweep 2 shows that increasing the number of independent
recordings does not, at this nuisance level, recover the true signal, but it does
**stabilize** the honest estimate (95% CI shrinks from ±0.13 at R=2 to ±0.03 at
R=32). Both honest evaluation and enough independent recordings are required.

### Study 2: DroneRF Leave-One-Recording-Out (LORO)

We evaluate on real DroneRF with leave-one-recording-out cross-validation. For
Type-ID we **pool the out-of-fold predictions across folds** and compute a single
macro-F1 with fixed labels (95% CI by bootstrap over the pooled predictions),
because each held-out recording is a single class.

**Type-ID (AR vs Bebop)** — independent recordings: 4 AR, 4 Bebop.

| Protocol | Macro-F1 [95% CI] | Accuracy [95% CI] |
|----------|-------------------|-------------------|
| Naive | 0.742 [0.728, 0.755] | 0.742 [0.729, 0.755] |
| **Honest (LORO, pooled)** | **0.455 [0.439, 0.470]** | 0.456 [0.441, 0.472] |

Honest identification falls to the two-class chance level: on an unseen flight the
spectral baseline cannot tell the two drones apart.

**Detection (drone vs background)** — *drone-side grouped, residual background
leakage* (DroneRF has a single independent background recording, so the negative
class cannot be grouped; treat as indicative).

| Protocol | ROC-AUC | Pd@1%FA |
|----------|---------|---------|
| Naive | 0.985 ± 0.005 | 0.764 ± 0.279 |
| Drone-side grouped (residual bg. leakage) | 0.978 ± 0.017 | 0.719 ± 0.275 |

### Study 3: Leakage-pathway Ablation

We tighten the grouping in three steps (pooled out-of-fold macro-F1, AR vs Bebop):

| Level | Macro-F1 |
|-------|----------|
| L0 — segment-level (no grouping) | 0.739 |
| L1 — file-level (band-halves separate) | 0.527 |
| L2 — recording-level (band-halves merged) | 0.524 |

Essentially all of the inflation is **segment-level** leakage within a recording
(L0 → L1); merging the two simultaneous band-halves of a flight changes the result
only marginally (L1 → L2). This is an attribution based on the dataset's known
structure (H and L are simultaneous captures of one flight), not a randomized
intervention.
