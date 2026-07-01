"""Dataset split helpers for leakage-aware evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold


def _assert_binary_partition(y_bin, indices, label):
    classes = set(np.asarray(y_bin)[indices].tolist())
    if classes != {0, 1}:
        raise AssertionError(f"{label} partition must contain both background and drone samples.")


def drone_side_grouped_detection_splits(y_bin, groups, n_splits=None, seed=0):
    """Yield detection splits with drone groups held out and background split by segment.

    DroneRF has only one independent background recording, so a fully grouped
    binary detection split cannot put background in both train and test. This
    helper leaves out whole drone recordings while intentionally splitting the
    background class at segment level. The residual background leakage is the
    documented detection protocol.
    """
    y_bin = np.asarray(y_bin)
    groups = np.asarray(groups)

    bg_indices = np.where(y_bin == 0)[0]
    drone_indices = np.where(y_bin == 1)[0]
    if len(bg_indices) == 0 or len(drone_indices) == 0:
        raise ValueError("Detection splits require both background and drone samples.")

    drone_groups = groups[drone_indices]
    unique_drone_groups = np.unique(drone_groups)
    if len(unique_drone_groups) < 2:
        raise ValueError("Detection splits require at least two independent drone groups.")

    if n_splits is None:
        n_splits = len(unique_drone_groups)
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(bg_indices) < 2:
        raise ValueError("Background segment splitting requires at least two background samples.")

    bg_split_count = min(n_splits, len(bg_indices))
    bg_cv = StratifiedKFold(n_splits=bg_split_count, shuffle=True, random_state=seed)
    bg_splits = list(bg_cv.split(bg_indices, np.zeros(len(bg_indices), dtype=int)))

    logo = LeaveOneGroupOut()
    for i, (dr_train_local, dr_test_local) in enumerate(
        logo.split(drone_indices, y_bin[drone_indices], drone_groups)
    ):
        if i >= n_splits:
            break

        bg_train_local, bg_test_local = bg_splits[i % len(bg_splits)]
        train_idx = np.concatenate((bg_indices[bg_train_local], drone_indices[dr_train_local]))
        test_idx = np.concatenate((bg_indices[bg_test_local], drone_indices[dr_test_local]))

        train_drone_groups = set(drone_groups[dr_train_local])
        test_drone_groups = set(drone_groups[dr_test_local])
        if train_drone_groups & test_drone_groups:
            raise AssertionError("Drone recording groups overlap between train and test.")

        _assert_binary_partition(y_bin, train_idx, "Train")
        _assert_binary_partition(y_bin, test_idx, "Test")
        yield train_idx, test_idx
