# 2. Concepts primer — the ideas, explained simply

This chapter assumes no background. Terms in **bold** are in the
[glossary](08_glossary.md).

## 2.1 What a "radio signature" is

A radio signal is an electromagnetic wave whose voltage wiggles over time. A
drone's radio link isn't a single pure tone; it has structure that depends on
*how* the drone and controller talk to each other:

- **Control link (uplink):** pilot commands sent up to the drone. To resist
  interference, many drones use **frequency-hopping spread spectrum (FHSS)** —
  the signal jumps rapidly between many narrow channels.
- **Video downlink:** a wide, continuous stream of imagery, often using
  **OFDM** (many sub-carriers at once), occupying a broad chunk of spectrum.
- **Telemetry:** smaller status packets.

Most consumer drones operate in the crowded **2.4 GHz** band (shared with Wi-Fi
and Bluetooth) and sometimes 5.8 GHz. The pattern of hops, bandwidths, and timing
forms an RF "fingerprint" that can distinguish a drone from background radio and,
sometimes, one drone model from another.

## 2.2 Capturing the signal: IQ data

A radio receiver doesn't store the raw multi-gigahertz wave. It shifts the band
of interest down and records two numbers per sample — the **in-phase (I)** and
**quadrature (Q)** components — which together describe the signal's amplitude and
phase. This **IQ data** is just a (complex-valued) time series, and everything
downstream is ordinary signal processing on it.

> In the DroneRF dataset, the 2.4 GHz band was captured by **two** receivers at
> once (a lower and an upper half) because the band is wider than one receiver
> can cover. Those two halves are part of the *same* recording — a fact that
> matters a great deal for honest evaluation (Chapter 5).

## 2.3 Turning the signal into a picture: the spectrum and spectrogram

The key question is "how much energy sits at each frequency?" The **Fourier
transform** answers it. Applied to short successive slices of the IQ stream, it
produces a **spectrogram**: time across the bottom, frequency up the side,
brightness for energy. The **power spectral density (PSD)** is the same idea
averaged over time — an energy-versus-frequency curve.

On these representations, a frequency-hopping control link appears as short
bright blips scattered across channels, while a video downlink appears as a wide,
steady band. Background Wi-Fi and Bluetooth have their own characteristic shapes.

## 2.4 Describing the spectrum with numbers: features

A classifier needs a compact, fixed-length set of numbers per segment. SpectraHawk
summarises the spectrum with interpretable **features**, including:

- **Band power** — how much energy is present, and where.
- **Spectral centroid** — the "centre of mass" frequency of the energy.
- **Spectral flatness** — whether the energy is tone-like (peaky) or noise-like
  (flat).
- **99% occupied bandwidth** — the width of spectrum containing almost all the
  energy (wide for video, narrow/hopping for control).

These mirror the quantities a radio engineer would read off a spectrum analyser.

## 2.5 Deciding what it is: classification

A **classifier** maps features to a label. SpectraHawk uses a **Random Forest**
(a voting committee of decision trees) as a transparent baseline. Two tasks:

- **Detection** — a binary decision: drone present or not. Quality is measured by
  the **detection rate** (fraction of real drones caught) versus the
  **false-alarm rate** (fraction of background wrongly flagged); the **ROC curve**
  plots one against the other and its area (**AUC**) is a single summary. Because
  false alarms are costly, we especially report the detection rate at a low (1%)
  false-alarm rate.
- **Type identification** — a multi-class decision (which drone). Quality is
  summarised by **macro-F1**, which averages performance across classes so a rare
  class can't be ignored.

## 2.6 The trap that makes everything look too good: data leakage

Here is the idea the whole project turns on. Suppose you cut one long flight
recording into many short segments and then shuffle all segments randomly into
"training" and "testing" piles. Segments from the *same* flight — same drone,
same room, same instant — now sit on **both** sides. A model can then succeed by
recognising *that particular recording's background* rather than the drone
itself, and its test score looks great but is **fake**: it won't survive a new
recording.

The fix is **recording-grouped evaluation**: keep every segment of one recording
entirely in training *or* entirely in testing, never split. Then the test score
reflects genuine generalisation to an unseen flight. As Chapter 5 shows, applying
this honestly to RF drone identification makes the numbers fall sharply — which
is the project's main finding.

➡ Next: [Architecture](03_architecture.md).
