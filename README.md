# SpectraHawk

**Passive RF Drone Detection and Identification**

SpectraHawk detects, classifies, and identifies Unmanned Aerial Vehicles (UAVs) using radio-frequency (RF) signatures. By analyzing the control, telemetry, and video links of drones, it can differentiate them from background interference (Wi-Fi, Bluetooth, thermal noise) and classify specific drone models.

## Dataset and Access
This repository is designed to run against the [DroneRF](https://www.kaggle.com/datasets/alishawang/dronerf) dataset (Al-Sa'd et al.). 

**Handling Data Leakage (Recording-Grouped Evaluation)**
A major flaw in RF machine learning evaluation is session-level data leakage—randomly splitting a single continuous recording across train and test sets guarantees artificially inflated metrics. SpectraHawk mitigates this by enforcing strict **Recording-Level Grouped Evaluation** on the drone classes. A flight recording session (BUI) used for training is never placed into the test set, yielding an honest measurement of real-world generalization.

*Note: The dataset groups High and Low frequency captures of the same flight into a single independent recording group.*

## Documentation

Full, beginner-friendly documentation is in [`docs/`](docs/README.md): a plain-language [overview](docs/01_overview.md) and [RF concepts primer](docs/02_concepts_primer.md), the [architecture](docs/03_architecture.md), the [detection & type-ID](docs/04_detection_and_typeid.md) chapter, the [evaluation & data-leakage story](docs/05_evaluation_rigor_and_leakage.md), a [getting-started guide](docs/06_getting_started.md), an [API reference](docs/07_api_reference.md), and a [glossary](docs/08_glossary.md). A paper-style write-up is in [`report/technical_note.md`](report/technical_note.md).

## Quickstart

The examples default to a fast internal **synthetic RF generator** (`spectrahawk.synth_rf`) so the quickstart stays lightweight even when a large DroneRF tree is present. Use `--real` to run against DroneRF.

```bash
# Install
python -m pip install -e .

# 1. Type ID quickstart
python examples/02_typeid_demo.py

# 2. Detection quickstart
python examples/01_detection_demo.py

# Real-data reproduction examples (slower; parses large CSV files)
python examples/02_typeid_demo.py --real --max-files 1
python examples/01_detection_demo.py --real --max-files 1
```

## Results

On a random forest baseline utilizing frequency-domain RF features (Band power, Spectral centroid, Flatness, 99% Bandwidth):

### 1. Honest Type-ID (AR vs. Bebop)
DroneRF contains 4 independent flight recordings each for the AR and Bebop drones, so Type-ID is our primary benchmark. We evaluate with **leave-one-recording-out** cross-validation, pooling the out-of-fold predictions across folds and computing one macro-F1 with fixed labels (95% CI by bootstrap). The model must therefore predict on an entirely unseen flight:

| Evaluation Split | Macro-F1 [95% CI] | Accuracy [95% CI] |
|---|---|---|
| Naive (Leaky) | 0.742 [0.728, 0.755] | 0.742 [0.729, 0.755] |
| **Recording-Grouped (Honest, LORO)** | **0.455 [0.439, 0.470]** | 0.456 [0.441, 0.472] |

*The naive split overstates Type-ID; under honest recording-grouped evaluation the baseline falls to the two-class chance level — simple spectral features do not generalize across unseen flight recordings. (Small N: 4 recordings/class.)*

### 2. Detection (Drone vs. Background)
**Caveat (Dataset Limitation):** DroneRF has a structural limitation—it effectively contains only a single independent background recording session (1 background, 4 AR, 4 Bebop, 1 Phantom). Because we cannot split a single session into both train and test folds without leakage, our detection split is **drone-side grouped only**. Background is split at the file/segment level, meaning there is residual negative-side leakage. Treat these detection numbers as indicative rather than a fully clean benchmark.

| Evaluation Split | ROC-AUC | Pd @ 1% FA |
|---|---|---|
| Naive (Leaky) | 0.985 ± 0.005 | 0.764 ± 0.279 |
| **Drone-Side Grouped (residual bg. leakage)** | **0.978 ± 0.017** | **0.719 ± 0.275** |

*Detection AUC barely changes because the single background recording must leak by necessity, and the 1%-false-alarm rate is highly variable across folds (±0.28). Detection is therefore indicative, not a clean benchmark; the honest signal in this dataset is the Type-ID collapse above.*

---
*(Note: A pivot to the DroneDetect dataset was attempted to resolve the background recording shortage, but the dataset is heavily login-walled and lacks any "drone-free" background class, forcing us back to DroneRF).*
