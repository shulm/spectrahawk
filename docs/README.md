# SpectraHawk Documentation

SpectraHawk is the radio-frequency (RF) companion to
[EchoHawk](https://github.com/shulm/echohawk). Where EchoHawk listens for a
drone's *sound*, SpectraHawk listens for a drone's *radio emissions* — the
control, telemetry, and video links between a drone and its operator — and uses
them to tell a drone apart from background radio activity and, where the data
allows, to identify which drone it is.

These docs are written so that **a reader with no signal-processing or
machine-learning background can follow the whole project**, while staying precise
enough for an engineer or reviewer.

## How to read these docs

**New to the topic?** Read in order:

1. [Overview](01_overview.md) — the problem and what SpectraHawk does, in plain language.
2. [Concepts primer](02_concepts_primer.md) — RF signals, drone radio links, spectra, and classification, explained simply.
3. [Architecture](03_architecture.md) — how the pieces fit together.

**Want the substance?**

4. [Detection & type identification](04_detection_and_typeid.md) — the tasks, the features, and the results.
5. [Evaluation, rigor & the leakage story](05_evaluation_rigor_and_leakage.md) — the most important chapter: how we kept the numbers honest, and what that revealed about RF drone benchmarks.

**Want to run or extend it?**

6. [Getting started](06_getting_started.md) — install and run every demo.
7. [API reference](07_api_reference.md) — the public functions.
8. [Glossary](08_glossary.md) — plain-language definitions of the jargon.

## One-paragraph summary

A drone and its controller exchange radio signals with a recognisable structure
(frequency-hopping control links, wide video downlinks). SpectraHawk turns a
captured RF segment into frequency-domain features and a spectrogram, and trains
a classifier to (a) separate drone activity from background radio and (b)
identify the drone model. Its defining feature is **honest evaluation**: it
groups the data by *independent recording* so a model is always tested on a
flight it never trained on. Doing this reveals that the near-perfect RF
drone-identification scores common in the literature are substantially inflated
by data leakage — under honest grouping, simple features barely beat chance. The
project runs on real data ([DroneRF](https://www.kaggle.com/datasets/alishawang/dronerf))
or, if absent, on a built-in synthetic RF generator.

See the [project README](../README.md) for headline results and the
[technical note](../report/technical_note.md) for the paper-style write-up.
