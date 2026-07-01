# 7. API reference

A guided tour of the public functions. Signatures are simplified; the docstrings
in each module are authoritative.

## `synth_rf.py`: synthetic RF

- **`generate_drone_rf(...)`** and **`generate_background_rf(...)`** produce
  synthetic RF segments.
- **`make_synthetic_dataset(...)`** returns `(X, y, groups, fs)` where `y=0` is
  background and positive labels are synthetic drone types. It is used as the
  download-free fallback and smoke-test data. The synthetic task is intentionally
  over-separable, so its scores are sanity checks, not benchmarks.

## `data_io.py`: real datasets

- **`DatasetNotFoundError`** is raised when a requested DroneRF tree contains no
  CSV files or no usable windows. The loader never exits the process.
- **`load_dronerf(root, ...)`** loads DroneRF raw windows, returning
  `(X_list, y_bin, y_type, groups, fs, groups_l1)`.
- **`load_dronerf_features(root, ...)`** streams DroneRF windows through a feature
  extractor and returns `(F, y_bin, y_type, groups, fs, groups_l1)`.
  `max_windows_per_file` can cap CSV parsing for demos; the default `None` reads
  all complete windows for reproduction.

Label scheme:

- `y_bin`: binary detection label, `0=no-drone/background`, `1=drone`.
- `y_type`: drone-model label, defined only where `y_bin == 1`; in the supported
  DroneRF studies `0=Bebop`, `1=AR`, `2=Phantom`. Background rows also carry `0`
  from the BUI code, so disambiguate background from Bebop with `y_bin`.
- `groups`: independent recording ID with simultaneous `H`/`L` band-halves
  merged.
- `groups_l1`: file/band-half level group before the `H`/`L` merge.

Background is handled explicitly: DroneRF contains a single independent
background recording, which is why detection is reported as drone-side grouped
with residual background leakage.

## `splits.py`: evaluation splits

- **`drone_side_grouped_detection_splits(y_bin, groups, ...)`** yields detection
  splits where drone recordings are group-disjoint and background is split at
  segment level. It asserts that both train and test contain both classes.

## `features.py`: feature extraction

- **`stft_rf(x, fs, ...)`** computes a complex RF short-time Fourier transform.
- **`psd_rf(x, fs, ...)`** computes power spectral density.
- **`feature_vector(x, fs)`** returns a fixed-length summary: band power,
  spectral centroid, spectral flatness, and 99% occupied bandwidth.
- **`feature_matrix_windows(X, fs, ...)`** vectorises feature extraction for a
  2-D window matrix.

## `models.py`: classifiers

- **`build_baseline(kind="rf", seed=0)`** returns a scikit-learn estimator.
  `kind="rf"` is a balanced Random Forest; `kind="svm"` is a scaled RBF SVM.
- **`features_matrix(X, fs, feature_fn)`** stacks `feature_fn` over many segments
  into a 2-D array.

## `evaluate.py`: metrics & plots

- **`pd_at_pfa(y_true, scores, pfa=0.01)`** estimates probability of detection at
  a fixed false-alarm rate.
- **`plot_roc(y_true, scores, path, title=...)`** saves an ROC figure and returns
  AUC.
- **`plot_confusion(y_true, y_pred, path, labels=...)`** saves a confusion matrix.
- **`plot_spectrogram(f, t, Zxx, path, title=...)`** saves an RF spectrogram.

The single most important detail is the `groups` array from `data_io`: it is
what makes the reported Type-ID numbers leakage-aware. See
[Chapter 5](05_evaluation_rigor_and_leakage.md).

Next: [Glossary](08_glossary.md).
