"""Detection demo: drone vs background.

The default run is a fast synthetic quickstart. Use ``--real`` to run against
DroneRF; the real-data grouped protocol leaves out drone recordings while
splitting the single background recording at segment level.
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from spectrahawk import data_io, evaluate, features, models  # noqa: E402
from spectrahawk.splits import drone_side_grouped_detection_splits  # noqa: E402
from spectrahawk.synth_rf import make_synthetic_dataset  # noqa: E402

DATA_DIR = os.path.join(_REPO, "data", "dronerf")
RESULTS_DIR = os.path.join(_REPO, "results")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="force the synthetic RF quickstart")
    parser.add_argument("--real", action="store_true", help="use DroneRF from data/dronerf")
    parser.add_argument("--max-files", type=int, default=None, help="maximum CSV files per DroneRF BUI")
    parser.add_argument("--fast", action="store_true", help="run a small quickstart-sized demo")
    parser.add_argument("--full", action="store_true", help="use the full default DroneRF cap")
    args = parser.parse_args()
    if args.synthetic and (args.real or args.max_files is not None or args.full):
        parser.error("--synthetic cannot be combined with real-data options")
    return args


def wants_fast(args):
    return args.fast or not args.full


def wants_real(args):
    return args.real or args.full or args.max_files is not None


def _assert_both_classes(y, indices, label):
    classes = set(np.asarray(y)[indices].tolist())
    if classes != {0, 1}:
        raise AssertionError(f"{label} partition must contain both classes; got {sorted(classes)}.")


def pd_at_fa_step(y_true, y_scores, fa_target=0.01):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    valid = np.where(fpr <= fa_target)[0]
    if len(valid) == 0:
        return 0.0
    return float(tpr[valid[-1]])


def metric_tuple(y_true, y_pred, y_scores):
    return (
        accuracy_score(y_true, y_pred),
        roc_auc_score(y_true, y_scores),
        pd_at_fa_step(y_true, y_scores, 0.01),
    )


def print_single_metrics(label, metrics):
    acc, auc, pd_1 = metrics
    print(f"[{label}] Accuracy : {acc:.4f}")
    print(f"[{label}] ROC-AUC  : {auc:.4f}")
    print(f"[{label}] Pd @ 1%FA: {pd_1:.4f}")
    print("-" * 30)


def print_cv_metrics(label, rows):
    rows = np.asarray(rows, dtype=float)
    names = ("Accuracy", "ROC-AUC", "Pd @ 1%FA")
    for i, name in enumerate(names):
        print(f"[{label}] {name}: {np.mean(rows[:, i]):.4f} +/- {np.std(rows[:, i]):.4f}")
    print("-" * 30)


def load_inputs(args):
    fast = wants_fast(args)
    if args.synthetic or not wants_real(args):
        n_per_class = 8 if fast else 100
        fs = 2e6 if fast else 20e6
        dur = 0.004 if fast else 0.01
        print("Generating synthetic RF data (fast quickstart path).")
        X, y, groups, fs = make_synthetic_dataset(
            n_per_class=n_per_class,
            fs=fs,
            dur=dur,
            n_classes=3,
            seed=42,
        )
        return X, (y > 0).astype(int), groups, fs, "synthetic", None

    max_files = args.max_files
    if max_files is None:
        max_files = 1 if fast else 5
    max_windows = 1 if fast else None

    try:
        print(
            f"Loading DroneRF from {DATA_DIR} "
            f"(max_files_per_bui={max_files}, max_windows_per_file={max_windows})..."
        )
        F, y, _, groups, fs, _ = data_io.load_dronerf_features(
            DATA_DIR,
            features.feature_vector,
            window_samples=200000,
            max_files_per_bui=max_files,
            max_windows_per_file=max_windows,
            batch_extract_fn=features.feature_matrix_windows,
        )
        first_window, *_ = next(
            data_io.iter_dronerf_windows(
                DATA_DIR,
                window_samples=200000,
                max_files_per_bui=max_files,
                max_windows_per_file=1,
            )
        )
        return [first_window], y, groups, fs, "real", F
    except data_io.DatasetNotFoundError as exc:
        print(f"DroneRF unavailable: {exc}")
        print("Run without --real for the synthetic quickstart, or install/download the data first.")
        raise SystemExit(2) from exc


def evaluate_single_split(F, y, label, grouped_splits=None):
    if grouped_splits is None:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.3,
            random_state=0,
            stratify=y,
        )
    else:
        train_idx, test_idx = next(grouped_splits)

    _assert_both_classes(y, train_idx, f"{label} train")
    _assert_both_classes(y, test_idx, f"{label} test")

    clf = models.build_baseline("rf", seed=0)
    clf.fit(F[train_idx], y[train_idx])
    scores = clf.predict_proba(F[test_idx])[:, 1]
    preds = clf.predict(F[test_idx])
    print_single_metrics(label, metric_tuple(y[test_idx], preds, scores))
    return y[test_idx], preds, scores


def evaluate_cv(F, y, groups, n_splits):
    class_counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(int(n_splits), int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("At least two folds with both classes are required.")

    naive_rows = []
    naive_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for train_idx, test_idx in naive_cv.split(F, y):
        _assert_both_classes(y, train_idx, "Naive train")
        _assert_both_classes(y, test_idx, "Naive test")
        clf = models.build_baseline("rf", seed=0)
        clf.fit(F[train_idx], y[train_idx])
        scores = clf.predict_proba(F[test_idx])[:, 1]
        preds = clf.predict(F[test_idx])
        naive_rows.append(metric_tuple(y[test_idx], preds, scores))

    grouped_rows = []
    pooled_y, pooled_preds, pooled_scores = [], [], []
    grouped_splits = drone_side_grouped_detection_splits(y, groups, n_splits=n_splits, seed=0)
    for train_idx, test_idx in grouped_splits:
        clf = models.build_baseline("rf", seed=0)
        clf.fit(F[train_idx], y[train_idx])
        scores = clf.predict_proba(F[test_idx])[:, 1]
        preds = clf.predict(F[test_idx])
        grouped_rows.append(metric_tuple(y[test_idx], preds, scores))
        pooled_y.append(y[test_idx])
        pooled_preds.append(preds)
        pooled_scores.append(scores)

    print_cv_metrics("Naive", naive_rows)
    print_cv_metrics("Drone-side grouped (residual bg. leakage)", grouped_rows)
    print("Class-presence assertions passed for every train/test partition.")

    return (
        np.concatenate(pooled_y),
        np.concatenate(pooled_preds),
        np.concatenate(pooled_scores),
    )


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y, groups, fs, source, F = load_inputs(args)

    print(f"Data source: {source}")
    print(f"Data: {len(y)} segments.")
    print(
        "Recordings (groups) by class: "
        f"Background={len(np.unique(groups[y == 0]))}, Drone={len(np.unique(groups[y == 1]))}"
    )

    print("Generating example spectrogram...")
    f_ax, t_ax, Zxx = features.stft_rf(X[0], fs=fs)
    evaluate.plot_spectrogram(f_ax, t_ax, Zxx, os.path.join(RESULTS_DIR, "spectrogram.png"))

    if F is None:
        print("Extracting features...")
        F = models.features_matrix(X, fs, features.feature_vector)
    else:
        print("Loaded streaming feature matrix.")

    if source == "real" and not wants_fast(args):
        n_splits = len(np.unique(groups[y == 1]))
        y_plot, preds_plot, scores_plot = evaluate_cv(F, y, groups, n_splits=n_splits)
    else:
        print("\n--- NAIVE SPLIT (segment-level) ---")
        evaluate_single_split(F, y, "Naive")

        print("\n--- DRONE-SIDE GROUPED SPLIT (residual background leakage) ---")
        grouped_splits = drone_side_grouped_detection_splits(y, groups, n_splits=2, seed=0)
        y_plot, preds_plot, scores_plot = evaluate_single_split(
            F,
            y,
            "Drone-side grouped",
            grouped_splits=grouped_splits,
        )
        print("Class-presence assertions passed for the grouped train/test partition.")

    evaluate.plot_roc(
        y_plot,
        scores_plot,
        title="RF Detection ROC (Drone-side grouped)",
        path=os.path.join(RESULTS_DIR, "roc.png"),
    )
    evaluate.plot_confusion(
        y_plot,
        preds_plot,
        labels=["Background", "Drone"],
        path=os.path.join(RESULTS_DIR, "confusion.png"),
    )

    print(f"Saved figures to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
