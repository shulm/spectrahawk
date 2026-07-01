# 6. Getting started

You do **not** need DroneRF to see SpectraHawk work. The public quickstart uses a
small synthetic RF dataset by default so it stays fast even if a large local
`data/dronerf/` tree exists.

## 6.1 Install

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"   # for tests
```

The core demos need `numpy`, `scipy`, `scikit-learn`, and `matplotlib`. The
optional DroneRF downloader needs the data extra:

```bash
python -m pip install -e ".[data]"
```

## 6.2 Run the fast demos

```bash
python examples/02_typeid_demo.py     # fast synthetic Type-ID quickstart
python examples/01_detection_demo.py  # fast synthetic detection quickstart
```

Useful flags:

```bash
python examples/01_detection_demo.py --synthetic
python examples/01_detection_demo.py --real --max-files 1
python examples/01_detection_demo.py --real --full
python examples/02_typeid_demo.py --real --max-files 1
```

`--synthetic` forces the no-data path. `--max-files N` caps CSV files per DroneRF
BUI. `--fast` keeps demos small; for real data it reads one window per CSV.
The default quickstart behavior is fast; `--full` uses the full default DroneRF
cap and reads all complete windows, so treat it as the reproduction path.

Figures are written to `results/`.

## 6.3 Run on real DroneRF

Download DroneRF from
[Kaggle](https://www.kaggle.com/datasets/alishawang/dronerf) into
`data/dronerf/`, or install the data extra and run:

```bash
python scripts/download_dronerf_subset.py --max-files 5
```

The CSV files are large. The loader groups DroneRF by independent recording
automatically, merging simultaneous `H`/`L` band-halves.

## 6.4 What each figure means

| Figure | How to read it |
|---|---|
| `spectrogram.png` | An example RF segment as a time-frequency picture. |
| `roc.png` | Detection rate vs false-alarm rate; higher and closer to the top-left is better. |
| `confusion.png` | Detection right/wrong counts; strong diagonal means better. |
| `typeid_confusion.png` | Drone-model confusion matrix under the grouped split. |

## 6.5 Run the tests

```bash
pytest -q
```

The smoke tests confirm the pipeline runs, the loader fails safely on missing
data, and the detection split helper preserves the documented class and grouping
invariants without the full dataset or a GPU.

## 6.6 Troubleshooting

- **"DroneRF unavailable"** is expected if `--real` is used before placing CSVs
  under `data/dronerf/`.
- **Downloader dependency missing** means install `python -m pip install -e
  ".[data]"`.
- **A download is blocked by a content filter** means fetch the dataset on an
  unrestricted machine and copy it into `data/dronerf/`. Do not disable
  certificate verification.

Next: [API reference](07_api_reference.md).
