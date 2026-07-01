"""Smoke tests for SpectraHawk."""
import os
import sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from spectrahawk import data_io, synth_rf, features, models
from spectrahawk.splits import drone_side_grouped_detection_splits
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

def test_synth_loader_shapes():
    """Test that the synthetic generator returns aligned arrays."""
    X, y, groups, fs = synth_rf.make_synthetic_dataset(
        n_per_class=6, fs=2e6, dur=0.004, n_classes=2, seed=0
    )
    assert len(X) == len(y)
    assert len(y) == len(groups)
    assert isinstance(X[0], np.ndarray)
    assert np.iscomplexobj(X[0])
    
def test_detection_synthetic():
    """Test that the baseline RF model can detect synthetic RF signatures."""
    X, y_orig, groups, fs = synth_rf.make_synthetic_dataset(
        n_per_class=8, fs=2e6, dur=0.004, n_classes=2, seed=0
    )
    y = (y_orig > 0).astype(int)
    
    F = models.features_matrix(X, fs, features.feature_vector)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
    train_idx, test_idx = next(gss.split(F, y, groups))
    
    clf = models.build_baseline("rf", seed=0)
    clf.fit(F[train_idx], y[train_idx])
    
    scores = clf.predict_proba(F[test_idx])[:, 1]
    auc = roc_auc_score(y[test_idx], scores)
    
    assert auc > 0.8, f"AUC {auc} is too low on synthetic data."


def test_downloader_imports_without_data_dependency():
    """The downloader module should not require kagglehub merely to import."""
    __import__("scripts.download_dronerf_subset")


def test_missing_dronerf_raises_typed_exception(tmp_path):
    """Dataset loading should raise, not terminate the Python process."""
    try:
        data_io.load_dronerf(str(tmp_path))
    except data_io.DatasetNotFoundError:
        pass
    else:
        raise AssertionError("Expected DatasetNotFoundError for an empty dataset directory.")


def test_drone_side_grouped_detection_split_invariants():
    """Tiny fixture for the residual-background-leakage detection protocol."""
    y = np.array([0] * 6 + [1] * 6)
    groups = np.array(["bg"] * 6 + ["d1"] * 2 + ["d2"] * 2 + ["d3"] * 2)

    splits = list(drone_side_grouped_detection_splits(y, groups, n_splits=3, seed=0))
    assert len(splits) == 3

    for train_idx, test_idx in splits:
        assert set(y[train_idx]) == {0, 1}
        assert set(y[test_idx]) == {0, 1}

        train_drone_groups = set(groups[train_idx][y[train_idx] == 1])
        test_drone_groups = set(groups[test_idx][y[test_idx] == 1])
        assert train_drone_groups.isdisjoint(test_drone_groups)
