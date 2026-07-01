# 8. Glossary

Plain-language definitions of the terms used in these docs.

**99% occupied bandwidth** — the width of spectrum containing 99% of a signal's
energy; wide for video downlinks, narrow/scattered for hopping control links.

**AUC (Area Under the ROC Curve)** — a single detector score, 0.5 (chance) to 1.0
(perfect).

**Background (no-drone)** — recorded radio activity with no drone present (Wi-Fi,
Bluetooth, noise); the negative class for detection.

**Band power** — how much signal energy lies within a frequency band.

**BUI (Binary Unique Identifier)** — DroneRF's label code for a recording,
encoding the drone and its flight mode (e.g. `10100`), with a trailing `H`/`L` for
the captured band-half.

**Classifier** — a function mapping features to a label (e.g. Random Forest).

**Control link (uplink)** — the radio channel carrying pilot commands to the
drone; often frequency-hopping.

**Counter-UAS** — defending against drones (Unmanned Aerial Systems).

**Data leakage** — when test information reaches the model during training,
inflating the apparent score. The central pitfall this project addresses.

**Detection rate (Pd)** — fraction of real drones correctly flagged.

**False-alarm rate (Pfa)** — fraction of background wrongly flagged as a drone;
kept low in security settings.

**Feature vector** — the fixed-length set of numbers describing one segment.

**FHSS (Frequency-Hopping Spread Spectrum)** — a scheme that rapidly hops a signal
across many narrow channels to resist interference; common in drone control links.

**Fourier transform / FFT** — the maths that converts a signal into its frequency
content.

**Grouped split (recording-grouped / GroupShuffleSplit)** — a train/test split
that keeps every segment of one recording on the same side, preventing leakage.

**IQ data** — the in-phase (I) and quadrature (Q) samples a receiver records; a
complex-valued time series representing the signal.

**Macro-F1** — the F1 score averaged equally across classes, so rare classes
count; the type-ID metric.

**OFDM (Orthogonal Frequency-Division Multiplexing)** — a wideband scheme using
many sub-carriers at once; common in drone video downlinks.

**Power Spectral Density (PSD)** — energy as a function of frequency, averaged
over time.

**Random Forest** — a classifier averaging many decision trees; SpectraHawk's
transparent baseline.

**Recording (flight) / session** — one continuous capture; the correct unit for
grouping. In DroneRF, the two band-halves of one flight are merged into a single
recording.

**RF (radio frequency)** — electromagnetic signals; here, a drone's radio
emissions.

**ROC curve** — detection rate vs false-alarm rate across all thresholds.

**Spectral centroid** — the energy-weighted "centre of mass" frequency of a
spectrum.

**Spectral flatness** — how tone-like (low) versus noise-like (high) a spectrum is.

**Spectrogram** — a picture of a signal: time across, frequency up, brightness =
energy.

**Type identification (Type-ID)** — classifying *which* drone model produced a
signal.

**UAS / UAV** — Unmanned Aerial System / Vehicle; a drone.

**Video downlink** — the radio channel streaming a drone's camera feed to the
operator; typically wideband.
