"""Download/copy a bounded DroneRF CSV subset with kagglehub.

The module is dependency-light on import. Install the data extra before running:

    python -m pip install -e ".[data]"
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

_SEGMENT_RE = re.compile(r"_(\d+)\.csv$", re.IGNORECASE)


def _segment_index(path: Path) -> int:
    match = _SEGMENT_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _load_kagglehub():
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise RuntimeError('kagglehub is required. Install it with: python -m pip install -e ".[data]"') from exc
    return kagglehub


def select_csv_subset(source_dir: Path, max_files_per_bui: int):
    csv_files = sorted(source_dir.rglob("*.csv"), key=lambda p: os.path.normcase(str(p.resolve())))
    if not csv_files:
        raise FileNotFoundError(f"No DroneRF CSV files found under {source_dir}.")

    by_bui = {}
    for path in csv_files:
        bui = path.name.split("_")[0]
        by_bui.setdefault(bui, {})
        by_bui[bui].setdefault(path.name, path)

    selected = []
    for bui in sorted(by_bui):
        files = sorted(
            by_bui[bui].values(),
            key=lambda p: (_segment_index(p), os.path.normcase(str(p.resolve()))),
        )
        selected.extend(files[:max_files_per_bui])
    return selected


def download_subset(max_files_per_bui=5, out_dir=None):
    kagglehub = _load_kagglehub()
    dataset = "alishawang/dronerf"
    repo_root = Path(__file__).resolve().parents[1]
    out_path = Path(out_dir) if out_dir is not None else repo_root / "data" / "dronerf"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading or locating Kaggle dataset cache: {dataset}")
    source_dir = Path(kagglehub.dataset_download(dataset))
    selected = select_csv_subset(source_dir, max_files_per_bui=max_files_per_bui)

    print(f"Selected {len(selected)} CSV files across {len({p.name.split('_')[0] for p in selected})} BUIs.")
    for i, src in enumerate(selected, start=1):
        dst = out_path / src.name
        print(f"Copying {i}/{len(selected)}: {src.name}")
        shutil.copy2(src, dst)

    print(f"Subset ready in {out_path}")
    return out_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-files", type=int, default=5, help="maximum CSV files per BUI")
    parser.add_argument("--out-dir", default=None, help="destination directory")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        download_subset(max_files_per_bui=args.max_files, out_dir=args.out_dir)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"Could not prepare DroneRF subset: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
