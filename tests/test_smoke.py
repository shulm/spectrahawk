"""Smoke tests for SpectraHawk."""
import os
import sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from spectrahawk import synth_rf, features, models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

def test_synth_loader_shapes():
    """Test that the synthetic generator returns aligned arrays."""
    X, y, groups, fs = synth_rf.make_synthetic_dataset(n_per_class=10, n_classes=2, seed=0)
    assert len(X) == len(y)
    assert len(y) == len(groups)
    assert isinstance(X[0], np.ndarray)
    assert np.iscomplexobj(X[0])
    
def test_detection_synthetic():
    """Test that the baseline RF model can detect synthetic RF signatures."""
    X, y_orig, groups, fs = synth_rf.make_synthetic_dataset(n_per_class=20, n_classes=2, seed=0)
    y = (y_orig > 0).astype(int)
    
    F = models.features_matrix(X, fs, features.feature_vector)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
    train_idx, test_idx = next(gss.split(F, y, groups))
    
    clf = models.build_baseline("rf", seed=0)
    clf.fit(F[train_idx], y[train_idx])
    
    scores = clf.predict_proba(F[test_idx])[:, 1]
    auc = roc_auc_score(y[test_idx], scores)
    
    assert auc > 0.8, f"AUC {auc} is too low on synthetic data."
