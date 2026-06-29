"""Machine Learning Models for SpectraHawk.

Includes baselines for RF signature detection and classification.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def build_baseline(kind="rf", seed=0):
    """Builds a scikit-learn baseline classifier.
    
    Parameters
    ----------
    kind : str
        "rf" (Random Forest) or "svm" (Support Vector Machine).
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    clf : sklearn estimator
    """
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    elif kind == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=seed, class_weight="balanced"))
        ])
    else:
        raise ValueError(f"Unknown baseline kind: {kind}")

def features_matrix(X_list, fs, extract_fn):
    """Extracts features for a list of clips and returns a matrix (N_samples, N_features)."""
    return np.stack([extract_fn(x, fs) for x in X_list])
