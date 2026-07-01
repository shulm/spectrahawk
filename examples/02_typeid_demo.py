"""Type-ID demo: which drone is it?

The default run is a fast synthetic quickstart. Use ``--real`` to run against
DroneRF with recording-level grouping.
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from spectrahawk import data_io, evaluate, features, models  # noqa: E402
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


def _assert_all_classes(y, indices, labels, split_name):
    present = set(np.asarray(y)[indices].tolist())
    expected = set(labels)
    if present != expected:
        raise AssertionError(f"{split_name} partition must contain labels {sorted(expected)}; got {sorted(present)}.")


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
        mask = y > 0
        X = [X[i] for i in range(len(X)) if mask[i]]
        y = y[mask] - 1
        groups = groups[mask]
        return X, y, groups, fs, ["Type 1", "Type 2"], "synthetic", None

    max_files = args.max_files
    if max_files is None:
        max_files = 1 if fast else 5
    max_windows = 1 if fast else None

    try:
        print(
            f"Loading DroneRF from {DATA_DIR} "
            f"(max_files_per_bui={max_files}, max_windows_per_file={max_windows})..."
        )
        F, y_bin, y, groups, fs, _ = data_io.load_dronerf_features(
            DATA_DIR,
            features.feature_vector,
            window_samples=200000,
            max_files_per_bui=max_files,
            max_windows_per_file=max_windows,
            batch_extract_fn=features.feature_matrix_windows,
        )
    except data_io.DatasetNotFoundError as exc:
        print(f"DroneRF unavailable: {exc}")
        print("Run without --real for the synthetic quickstart, or install/download the data first.")
        raise SystemExit(2) from exc

    # y_type is a model label only where y_bin == 1: 0=Bebop, 1=AR, 2=Phantom.
    mask = (y_bin == 1) & (y <= 1)
    F = F[mask]
    y = y[mask]
    groups = groups[mask]
    return [], y, groups, fs, ["Bebop", "AR"], "real", F


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y, groups, fs, classes, source, F = load_inputs(args)
    labels = np.arange(len(classes))

    print(f"Data source: {source}")
    print(f"Data: {len(y)} drone segments.")
    print(f"Recordings (groups): {len(np.unique(groups))}")
    print("\n[CAVEAT] With very few independent recordings, grouped type-ID is statistically thin.")
    print("Results are indicative of pipeline functioning, not definitive real-world performance.\n")

    if F is None:
        print("Extracting features...")
        F = models.features_matrix(X, fs, features.feature_vector)
    else:
        print("Loaded streaming feature matrix.")

    print("\n--- NAIVE SPLIT (segment-level) ---")
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.3,
        random_state=0,
        stratify=y,
    )
    _assert_all_classes(y, train_idx, labels, "Naive train")
    _assert_all_classes(y, test_idx, labels, "Naive test")
    clf_naive = models.build_baseline("rf", seed=0)
    clf_naive.fit(F[train_idx], y[train_idx])
    preds_n = clf_naive.predict(F[test_idx])
    print(f"[Naive] Accuracy : {accuracy_score(y[test_idx], preds_n):.4f}")
    print(f"[Naive] Macro-F1 : {f1_score(y[test_idx], preds_n, labels=labels, average='macro'):.4f}")
    print("------------------------------\n")

    print("--- GROUPED SPLIT (recording-level) ---")
    gss = GroupShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
    for train_idx, test_idx in gss.split(F, y, groups):
        try:
            _assert_all_classes(y, train_idx, labels, "Grouped train")
            _assert_all_classes(y, test_idx, labels, "Grouped test")
            break
        except AssertionError:
            continue
    else:
        raise AssertionError("Could not find a grouped split containing every class in train and test.")

    if set(groups[train_idx]) & set(groups[test_idx]):
        raise AssertionError("Recording groups overlap between train and test.")

    clf = models.build_baseline("rf", seed=0)
    clf.fit(F[train_idx], y[train_idx])
    preds = clf.predict(F[test_idx])
    acc = accuracy_score(y[test_idx], preds)
    macro_f1 = f1_score(y[test_idx], preds, labels=labels, average="macro")

    print(f"[Grouped] Accuracy : {acc:.4f}")
    print(f"[Grouped] Macro-F1 : {macro_f1:.4f}\n")
    print("Class-presence and group-disjointness assertions passed.")

    present_labels = np.unique(np.concatenate((y[test_idx], preds)))
    target_names = [classes[int(i)] for i in present_labels]

    print("Grouped Classification Report:")
    print(classification_report(y[test_idx], preds, labels=present_labels, target_names=target_names))

    evaluate.plot_confusion(
        y[test_idx],
        preds,
        labels=target_names,
        path=os.path.join(RESULTS_DIR, "typeid_confusion.png"),
    )

    print(f"Saved figure to {RESULTS_DIR}/typeid_confusion.png")


if __name__ == "__main__":
    main()
