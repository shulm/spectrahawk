# 5. Evaluation, rigor & the leakage story

This is the most important chapter. SpectraHawk's headline is not a score: it is
a demonstration of how RF drone benchmarks mislead when evaluated carelessly, and
how to do it honestly.

## 5.1 How we measure

- **Train/test split.** Fit on one part of the data, measure on a separate part.
  Measuring on training data proves nothing.
- **Detection metrics.** ROC-AUC summarises the detection-versus-false-alarm
  trade-off; we also report the probability of detection at fixed false-alarm
  rates.
- **Type-ID metric.** Macro-F1 averages per-class performance so an
  over-represented class cannot hide poor results on a rare one.

## 5.2 Leakage in RF data

**Data leakage** is when information from the test set reaches the model during
training, inflating the apparent score. In segmented RF recordings it arises in
two specific ways, both of which SpectraHawk controls:

1. **Segment leakage.** One continuous recording is cut into many short segments.
   If those segments are shuffled freely, near-identical slices of one flight land
   in both train and test, and the model can recognise the recording instead of
   the drone.
2. **Band-half leakage (specific to DroneRF).** The 2.4 GHz band was captured by
   two receivers, a low half (`L`) and a high half (`H`), simultaneously from the
   same flight. Treating `H` and `L` as two independent recordings would let the
   same flight's two halves straddle the split. SpectraHawk merges `H` and `L` of
   a flight into one recording group, closing this subtler leak.

The remedy for both is **recording-grouped cross-validation**: the set of
recordings in training and the set in testing are disjoint, so a model is always
judged on a flight it has never seen.

## 5.3 What DroneRF actually contains

After merging band-halves, the number of independent recordings is small:

| Class | Independent recordings |
|---|---|
| Background (no drone) | **1** |
| AR drone | 4 |
| Bebop drone | 4 |
| Phantom drone | **1** |

Two consequences follow directly:

- **Honest detection is not fully possible.** With a single background recording,
  you cannot place background in both train and test without leakage. Detection is
  therefore reported drone-side-grouped with residual background leakage and
  labelled indicative.
- **Honest identification is limited to AR vs Bebop.** Only these two drones have
  enough independent recordings to hold one out. Phantom (one recording) cannot be
  tested under grouping and is excluded from the grouped metric.

## 5.4 The honest numbers and the inflation they expose

| Task | Naive (leaky) | Honest/grouped protocol |
|---|---|---|
| Type-ID (AR vs Bebop), Macro-F1 | 0.742 [0.728, 0.755] | **0.455 [0.439, 0.470]** |
| Type-ID (AR vs Bebop), Accuracy | 0.742 [0.729, 0.755] | **0.456 [0.441, 0.472]** |
| Detection, ROC-AUC | 0.985 +/- 0.005 | **0.978 +/- 0.017** (drone-side grouped, residual bg. leakage) |
| Detection, Pd @ 1% FA | 0.764 +/- 0.279 | **0.719 +/- 0.275** (drone-side grouped, residual bg. leakage) |

The naive splits look impressive; the honest splits do not. Type identification
falls to the two-class chance level. Detection remains only indicative because
the single background recording must be split at segment level, leaving residual
negative-side leakage. Much of the apparent identification performance was the
model recognising specific recordings.

## 5.5 Why this matters

Published RF drone-identification results frequently report accuracies above 95%.
SpectraHawk's analysis suggests that a large part of such figures can be
**recording memorisation rather than drone recognition**, and that a small dataset
like DroneRF, with only a few independent captures per drone, cannot support a
confident identification claim at all. For anyone building or evaluating an RF
counter-UAS system, the lesson is concrete: insist on recording-grouped
evaluation and on enough independent captures, or the reported performance will
not survive contact with a new environment.

This is the same discipline applied in the acoustic sister project, EchoHawk;
together the two show that the pitfall is general across drone-signature
modalities.

Next: [Getting started](06_getting_started.md).
