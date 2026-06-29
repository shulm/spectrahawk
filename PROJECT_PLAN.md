# SpectraHawk — RF-Based Drone Detection & Identification (Project 2)

**The RF sibling to EchoHawk.** Where EchoHawk listens for a drone's acoustic
blade-passing signature, SpectraHawk detects and *identifies* drones from their
**radio-frequency signature** (the controller/telemetry link). Acoustic + RF are
the two dominant **passive** counter-UAS modalities, so this pair shows command
of the whole passive-sensing landscape — directly relevant to Insignito, which
markets acoustic detection precisely for drones with *minimal RF* signature.

## Why this is the right second project
- **New modality, new task.** Not a repeat of EchoHawk: this adds EM-wave signal
  processing and a richer task — drone **type** and **flight-mode** identification,
  not just binary detection.
- **Same rigorous core, reused.** STFT/spectrogram features → classical + CNN,
  evaluated with the **leakage-aware grouped split + Pd@FA** harness we built for
  EchoHawk. Reusing that methodology is itself a strong signal of maturity.
- **Physics fit.** RF is EM-wave propagation (bandwidth occupancy, frequency
  hopping, modulation) — aligns with the JD's "acoustic/thermal/optical" framing
  and your wave background.

## Datasets (Kaggle)
- **Primary — Noisy Drone RF Signal Classification v2** (`sgluege`): purpose-built
  ML benchmark; RF signal vectors with controlled injected noise (Bluetooth /
  Wi-Fi / amplifier + Gaussian). The built-in noise enables a **robustness-vs-SNR**
  story mirroring EchoHawk's DOA-vs-SNR curve.
- **Secondary — DroneRF** (Al-Sa'd et al.): 227 RF segments, 3 drones
  (Bebop / AR / Phantom) + background; supports binary detection, 4-class type,
  and 10-class flight-mode identification.
- **Access caveat:** Kaggle sits behind the Rimon filter (as HuggingFace did) and
  its API uses a CDN. The doer probes reachability first; if blocked, download on
  an unblocked machine or via `kagglehub`, and the loader must work from a local
  folder regardless.

## Scope
1. **Detection** — drone vs. background RF (binary).
2. **Type ID** — which drone (multi-class).
3. **Mode ID** (stretch) — flight mode (DroneRF 10-class).
4. **Robustness** — performance vs. injected-noise level.

## Methods
- Features: RF **spectrogram (STFT)**, power spectral density, spectral
  statistics, bandwidth-occupancy / frequency-hopping cues.
- Models: classical baseline (RF/SVM on spectral features) + a CNN on RF
  spectrograms (reuse the EchoHawk CNN pattern; GPU via `conda activate rl`).
- Evaluation: **grouped split by source recording/segment** (no segment leakage),
  Pd@1/5/10% FA for detection, multi-class accuracy + confusion for ID, and a
  noise-robustness curve. Report naive vs. grouped where relevant.

## Deliverables (mirror EchoHawk)
Python package + runnable `examples/` + `tests/` + GitHub Actions CI + README +
an arXiv-style technical note. Published to `github.com/shulm/spectrahawk`.

## Methodology guardrails (carried over from EchoHawk)
- Leakage-aware grouped evaluation from **day one**.
- Honest measured numbers only; separate naive vs. grouped tables.
- **Never** disable TLS/SSL verification. No raw data or model weights in git.
- Fixed seeds; reproducible.

## Milestones
- **M1** — scaffold + RF data loader (+ reachability probe) + baseline **detection
  and type-ID** with the grouped-split eval + figures.
- **M2** — CNN on RF spectrograms + **robustness-vs-noise** curve; mode-ID stretch.
- **M3** — tests/CI, README, technical note, publish.
