# 4. Detection & type identification

This chapter explains the two tasks, the features behind them, the datasets, and
the results. For the underlying ideas, see the
[concepts primer](02_concepts_primer.md).

## 4.1 From RF segment to features

Each captured segment is divided into fixed windows. For every window
(`features.py`):

- a **spectrogram** and **power spectral density** characterise the energy across
  frequency and time;
- a compact **feature vector** summarises the spectrum with **band power**,
  **spectral centroid**, **spectral flatness**, and **99% occupied bandwidth**.

These capture what separates a frequency-hopping control link, a wide video
downlink, and ordinary background radio.

## 4.2 The classifier

The baseline is a **Random Forest** (300 trees) over the feature vector: fast,
transparent, and a fair reference point. The same model serves both tasks; only
the labels and the evaluation grouping change.

## 4.3 Datasets

- **DroneRF** (Al-Sa'd et al.) recorded 2.4 GHz RF for three drones (AR drone,
  Bebop drone, Phantom) plus a background (no-drone) recording. Each recording is
  labelled by a **BUI** (a code for drone + flight mode) and captured in two
  band-halves (`H`/`L`). After accounting for the band-halves, the dataset
  contains only a handful of *independent* recordings per class (see
  [Chapter 5](05_evaluation_rigor_and_leakage.md)).
- **Synthetic RF** (`synth_rf.py`) creates narrowband bursts, frequency hopping,
  and background, used as a runnable fallback and a pipeline sanity check. It is
  over-separable by design, so its scores are not meaningful benchmarks.

## 4.4 Results

All numbers are reported **two ways**: a naive split (segments shuffled freely)
and an honest split (grouped by independent recording).

### Type identification: AR vs Bebop (the honest headline)

DroneRF has four independent recordings each for the AR and Bebop drones, enough
to hold whole recordings out, so this is the project's primary leakage-free
benchmark. Phantom has only one recording and cannot be held out, so it is
excluded from the grouped metric.

| Evaluation split | Macro-F1 [95% CI] | Accuracy [95% CI] |
|---|---|---|
| Naive (leaky) | 0.742 [0.728, 0.755] | 0.742 [0.729, 0.755] |
| **Recording-grouped (honest, LORO)** | **0.455 [0.439, 0.470]** | **0.456 [0.441, 0.472]** |

On an entirely unseen flight recording, the baseline's spectral features barely
separate the two drones (near the two-class chance level). The naive split
overstates the real ability by recognising the specific recording. This is a
leave-one-recording-out result over only four recordings per class, so treat the
confidence intervals as conditional on a very small independent-recording pool.

### Detection: drone vs background

| Evaluation split | ROC-AUC | Pd @ 1% FA |
|---|---|---|
| Naive (leaky) | 0.985 +/- 0.005 | 0.764 +/- 0.279 |
| **Drone-side grouped (residual bg. leakage)** | **0.978 +/- 0.017** | **0.719 +/- 0.275** |

Detection looks strong, but DroneRF contains only **one** independent background
recording, so the negative class cannot be split without leakage. The honest
split therefore groups only the drone (positive) side; the background side still
leaks. These numbers are **indicative, not a clean benchmark**. The honest signal
in DroneRF is the Type-ID collapse above; detection remains limited by the single
background recording.

Next: [Evaluation, rigor & the leakage story](05_evaluation_rigor_and_leakage.md).
