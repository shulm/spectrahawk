"""Data I/O for SpectraHawk RF datasets."""
import glob
import os
import re

import numpy as np


_SEGMENT_RE = re.compile(r"_(\d+)\.csv$", re.IGNORECASE)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a requested dataset tree contains no usable DroneRF windows."""


def _segment_index(path: str) -> int:
    match = _SEGMENT_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else -1


def _discover_dronerf_csvs(root: str, max_files_per_bui=5):
    csv_files = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise DatasetNotFoundError(f"No DroneRF .csv files found in {root}.")

    # Group by BUI, de-duplicate repeated extractions by basename, then cap
    # deterministically so repeated runs use the same recording segments.
    bui_dict = {}
    for fpath in sorted(csv_files, key=lambda p: os.path.normcase(os.path.abspath(p))):
        basename = os.path.basename(fpath)
        bui = basename.split("_")[0]
        bui_dict.setdefault(bui, {})
        bui_dict[bui].setdefault(basename, fpath)

    selected = []
    for bui in sorted(bui_dict):
        files = sorted(
            bui_dict[bui].values(),
            key=lambda p: (_segment_index(p), os.path.normcase(os.path.abspath(p))),
        )
        selected.extend(files[:max_files_per_bui])

    return selected


def _metadata_from_path(fpath: str):
    basename = os.path.basename(fpath)
    parts = basename.split("_")
    if len(parts) < 2:
        return None

    bui = parts[0]  # e.g. 10100H
    if len(bui) < 5:
        return None

    type_map = {"00": 0, "01": 1, "10": 2, "11": 3}
    is_drone = int(bui[0])
    drone_type_str = bui[1:3]
    d_type = type_map.get(drone_type_str, 0)
    group_l1 = bui
    group_id = bui[:-1] if bui[-1] in ("H", "L") else bui
    return is_drone, d_type, group_id, group_l1


def iter_dronerf_windows(
    root: str,
    window_samples=200000,
    max_files_per_bui=5,
    max_windows_per_file=None,
):
    """Yield DroneRF windows and labels without retaining whole CSV files."""
    csv_files = _discover_dronerf_csvs(root, max_files_per_bui=max_files_per_bui)

    for fpath in csv_files:
        metadata = _metadata_from_path(fpath)
        if metadata is None:
            continue

        is_drone, d_type, group_id, group_l1 = metadata
        count = -1 if max_windows_per_file is None else int(window_samples * max_windows_per_file)
        vals = np.fromfile(fpath, dtype=np.float32, sep=",", count=count)
        iq = vals.astype(np.complex64, copy=False)

        for i in range(0, len(iq) - window_samples + 1, window_samples):
            window = iq[i:i + window_samples]
            yield window, is_drone, d_type, group_id, group_l1


def load_dronerf_features(
    root: str,
    extract_fn,
    window_samples=200000,
    max_files_per_bui=5,
    max_windows_per_file=None,
    batch_extract_fn=None,
):
    """Load DroneRF as feature rows, streaming windows through ``extract_fn``."""
    X_rows = []
    y_binary_list = []
    y_type_list = []
    groups_list = []
    groups_l1_list = []
    fs = 20e6

    if batch_extract_fn is not None:
        for fpath in _discover_dronerf_csvs(root, max_files_per_bui=max_files_per_bui):
            metadata = _metadata_from_path(fpath)
            if metadata is None:
                continue

            is_drone, d_type, group_id, group_l1 = metadata
            count = -1 if max_windows_per_file is None else int(window_samples * max_windows_per_file)
            vals = np.fromfile(fpath, dtype=np.float32, sep=",", count=count)
            iq = vals.astype(np.complex64, copy=False)
            n_windows = len(iq) // window_samples
            if n_windows == 0:
                continue

            windows = iq[:n_windows * window_samples].reshape(n_windows, window_samples)
            X_rows.append(batch_extract_fn(windows, fs))
            y_binary_list.extend([is_drone] * n_windows)
            y_type_list.extend([d_type] * n_windows)
            groups_list.extend([group_id] * n_windows)
            groups_l1_list.extend([group_l1] * n_windows)

        if not X_rows:
            raise DatasetNotFoundError(f"No usable DroneRF windows found in {root}.")

        return (
            np.vstack(X_rows),
            np.array(y_binary_list),
            np.array(y_type_list),
            np.array(groups_list),
            fs,
            np.array(groups_l1_list),
        )

    for window, is_drone, d_type, group_id, group_l1 in iter_dronerf_windows(
        root,
        window_samples=window_samples,
        max_files_per_bui=max_files_per_bui,
        max_windows_per_file=max_windows_per_file,
    ):
        X_rows.append(extract_fn(window, fs))
        y_binary_list.append(is_drone)
        y_type_list.append(d_type)
        groups_list.append(group_id)
        groups_l1_list.append(group_l1)

    if not X_rows:
        raise DatasetNotFoundError(f"No usable DroneRF windows found in {root}.")

    return (
        np.stack(X_rows),
        np.array(y_binary_list),
        np.array(y_type_list),
        np.array(groups_list),
        fs,
        np.array(groups_l1_list),
    )

def load_dronerf(root: str, window_samples=200000, max_files_per_bui=5, max_windows_per_file=None):
    """Loads DroneRF dataset (Al-Sa'd et al.)
    
    Expects CSV files where filename is {BUI}_{segment}.csv
    BUI format:
      1st char: 0=Background, 1=Drone
      2nd/3rd chars: 00=Bebop when y_binary=1 (or no-drone metadata
      when y_binary=0), 01=AR, 10=Phantom
      4th/5th chars: Mode
      6th char: Band (H or L)
      
    Segments each CSV into fixed windows of `window_samples`. Set
    `max_windows_per_file` for fast demos; leave it as `None` for full
    reproduction behavior.
    
    Returns
    -------
    X_list : list of np.ndarray (complex RF data)
    y_binary : np.ndarray (0=no-drone/background, 1=drone)
    y_type : np.ndarray
        Drone-model label, defined for rows where ``y_binary == 1``:
        0=Bebop, 1=AR, 2=Phantom. Background rows also carry 0 from
        the DroneRF BUI code and must be disambiguated with ``y_binary``.
    groups : np.ndarray (string or int IDs for the parent recording)
    fs : float (Mocked at 20MHz or 40MHz, usually unspecified but we assume 20Mhz for feature extraction)
    """
    X_list = []
    y_binary_list = []
    y_type_list = []
    groups_list = []
    groups_l1_list = []
    
    # DroneRF was captured at high sample rates, we'll assume 20 MHz
    fs = 20e6
    
    for window, is_drone, d_type, group_id, group_l1 in iter_dronerf_windows(
        root,
        window_samples=window_samples,
        max_files_per_bui=max_files_per_bui,
        max_windows_per_file=max_windows_per_file,
    ):
        X_list.append(window.copy())
        y_binary_list.append(is_drone)
        y_type_list.append(d_type)
        groups_list.append(group_id)
        groups_l1_list.append(group_l1)

    if not X_list:
        raise DatasetNotFoundError(f"No usable DroneRF windows found in {root}.")
            
    return X_list, np.array(y_binary_list), np.array(y_type_list), np.array(groups_list), fs, np.array(groups_l1_list)
