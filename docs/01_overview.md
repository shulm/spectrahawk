# 1. Overview — what SpectraHawk is and why it exists

## The problem

Protecting a site from drones (the field of **counter-UAS**, counter–Unmanned
Aerial Systems) means detecting them early and, ideally, identifying what they
are. Most consumer and commercial drones stay in constant radio contact with
their operator: a **control link** carries pilot commands up to the drone, a
**telemetry link** carries status back, and many drones stream **video** down to
the controller. Those transmissions are a signature you can listen for with a
radio receiver — no line of sight, no daylight, and (because you are only
listening) nothing that gives your own position away.

This is the **RF (radio-frequency)** approach to counter-UAS. Its strengths are
range and the richness of the signal; its main weakness is that a fully
autonomous drone flying a pre-programmed mission may emit little or nothing. RF
is therefore complementary to acoustic sensing (see the sister project,
EchoHawk), which works precisely when a drone is quiet on the radio.

SpectraHawk is a clean, well-documented reference implementation of the RF
approach — a demonstrator of the core methods and, above all, of how to evaluate
them honestly.

## What SpectraHawk does

1. **Detect** — given a captured RF segment, decide whether it contains drone
   activity or only background radio (Wi-Fi, Bluetooth, noise).
2. **Identify** — classify *which* drone model produced the signal.
3. **Evaluate honestly** — measure both tasks in a way that does not let the
   model cheat by memorising a particular recording (the heart of the project).

## What's in the box

- A **library** (`spectrahawk/`) with the feature extraction, models, data
  loaders, and evaluation utilities.
- Runnable **demos** (`examples/`) that reproduce the figures.
- A **synthetic RF generator** so the pipeline runs with no download.
- A loader for the real [DroneRF](https://www.kaggle.com/datasets/alishawang/dronerf)
  dataset, with leakage-safe recording grouping.
- Tests, continuous integration, and a Docker file.

## Headline results (plain language)

- **Identifying a drone is hard when you test honestly.** Telling an AR drone
  from a Bebop, when the model must judge an *entirely unseen* flight recording,
  reaches only ~57% balanced score — barely above the 50% coin-flip. A naive
  evaluation that lets recordings leak across the split reports ~72%, overstating
  the real ability.
- **Detection** (drone vs background) looks strong (~94% accuracy) but rests on a
  dataset with only one background recording, so its honest interpretation is
  limited (see the [evaluation chapter](05_evaluation_rigor_and_leakage.md)).

## The honest takeaway

SpectraHawk's most valuable result is not a high score — it is the demonstration
that **standard RF drone-identification benchmarks are inflated by data
leakage**, and that the public DroneRF dataset contains too few independent
recordings to support a confident identification claim. That is a useful,
transferable lesson for anyone building or buying RF counter-UAS systems.
