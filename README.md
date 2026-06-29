# SpectraHawk

**Passive RF Drone Detection and Identification**

SpectraHawk detects, classifies, and identifies Unmanned Aerial Vehicles (UAVs) using radio-frequency (RF) signatures. By analyzing the control, telemetry, and video links of drones, it can differentiate them from background interference (Wi-Fi, Bluetooth, thermal noise) and classify specific drone models.

## Dataset and Access
This repository is designed to run against the [DroneRF](https://www.kaggle.com/datasets/alishawang/dronerf) dataset (Al-Sa'd et al.). 

**Handling Data Leakage (Recording-Grouped Evaluation)**
A major flaw in RF machine learning evaluation is session-level data leakage—randomly splitting a single continuous recording across train and test sets guarantees artificially inflated metrics. SpectraHawk mitigates this by enforcing strict **Recording-Level Grouped Evaluation** on the drone classes. A flight recording session (BUI) used for training is never placed into the test set, yielding an honest measurement of real-world generalization.

*Note: The dataset groups High and Low frequency captures of the same flight into a single independent recording group.*

## Quickstart

If the dataset is missing, the examples will fall back to an internal **synthetic RF generator** (`spectrahawk.synth_rf`) which simulates narrowband bursts, frequency hopping, and background interference.

```bash
# 1. Type ID (AR vs Bebop) - Primary Benchmark
python examples/02_typeid_demo.py

# 2. Detection (Drone vs Background)
python examples/01_detection_demo.py
```

## Results

On a random forest baseline utilizing frequency-domain RF features (Band power, Spectral centroid, Flatness, 99% Bandwidth):

### 1. Honest Type-ID (AR vs. Bebop)
Because DroneRF contains exactly 4 independent flight recordings for both the AR and Bebop drones, Type-ID serves as our **primary, fully leakage-free benchmark**. When grouped strictly by flight recording (so the model must predict on an entirely unseen flight):

| Evaluation Split | Macro-F1 | Accuracy |
|---|---|---|
| Naive (Leaky) | 71.8% | 71.8% |
| **Flight-Grouped (Honest)** | **57.2%** | **57.4%** |

*The naive split traditionally overestimates Type-ID. In reality, simple spectral features struggle to generalize across entirely unseen flight recordings. This highlights the real-world difficulty of RF drone classification.*

### 2. Detection (Drone vs. Background)
**Caveat (Dataset Limitation):** DroneRF has a structural limitation—it effectively contains only a single independent background recording session (1 background, 4 AR, 4 Bebop, 1 Phantom). Because we cannot split a single session into both train and test folds without leakage, our detection split is **drone-side grouped only**. Background is split at the file/segment level, meaning there is residual negative-side leakage. Treat these detection numbers as indicative rather than a fully clean benchmark.

| Evaluation Split | Accuracy | ROC-AUC | Pd @ 1% FA |
|---|---|---|---|
| Naive (Leaky) | 95.67% | 0.983 | 84.5% |
| **Drone-Side Grouped (Honest Positives)** | **93.89%** | **0.937** | **69.2%** |

*Notice how the probability of detection at a 1% false alarm rate collapses when we enforce strict flight recording isolation on the positive (drone) class!*

---
*(Note: A pivot to the DroneDetect dataset was attempted to resolve the background recording shortage, but the dataset is heavily login-walled and lacks any "drone-free" background class, forcing us back to DroneRF).*
