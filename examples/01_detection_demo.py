"""Detection Demo: Drone vs Background (Binary).

Loads DroneRF subset, extracts features, and runs Random Forest.
Shows naive (clip-level) vs grouped (recording-level) splitting to demonstrate leakage.
"""
import os
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from spectrahawk import data_io, features, models, evaluate
from spectrahawk.synth_rf import make_synthetic_dataset

DATA_DIR = os.path.join(_REPO, "data", "dronerf")
RESULTS_DIR = os.path.join(_REPO, "results")

def evaluate_model(clf, F_train, y_train, F_test, y_test, label):
    clf.fit(F_train, y_train)
    scores = clf.predict_proba(F_test)[:, 1]
    preds = clf.predict(F_test)
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, scores)
    pd_1 = evaluate.pd_at_pfa(y_test, scores, pfa=0.01)
    pd_5 = evaluate.pd_at_pfa(y_test, scores, pfa=0.05)
    pd_10 = evaluate.pd_at_pfa(y_test, scores, pfa=0.10)
    
    print(f"[{label}] Accuracy : {acc:.4f}")
    print(f"[{label}] ROC-AUC  : {auc:.4f}")
    print(f"[{label}] Pd @ 1%FA: {pd_1:.4f}")
    print(f"[{label}] Pd @ 5%FA: {pd_5:.4f}")
    print(f"[{label}] Pd @10%FA: {pd_10:.4f}")
    print("-" * 30)
    return preds, scores

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0 and 'README.md' not in os.listdir(DATA_DIR)[0]:
        print("Loading real DroneRF subset...")
        X, y, y_type, groups, fs = data_io.load_dronerf(DATA_DIR, window_samples=200000)
    else:
        print("Real dataset not found. Generating synthetic RF data (pipeline sanity check - over-separable by design)...")
        X, y, groups, fs = make_synthetic_dataset(n_per_class=100, fs=20e6, n_classes=3, seed=42)
        y = (y > 0).astype(int)
        
    print(f"Data: {len(X)} segments.")
    print(f"Recordings (groups) by class: Background={len(np.unique(groups[y==0]))}, Drone={len(np.unique(groups[y==1]))}")
    
    # Save one spectrogram
    print("Generating example spectrogram...")
    f_ax, t_ax, Zxx = features.stft_rf(X[0], fs=fs)
    evaluate.plot_spectrogram(f_ax, t_ax, Zxx, os.path.join(RESULTS_DIR, "spectrogram.png"))
    
    print("Extracting features...")
    F = models.features_matrix(X, fs, features.feature_vector)
    
    # --- Naive Split (Clip-level, Leakage) ---
    print("\n--- NAIVE SPLIT (File/Clip-level) ---")
    F_tr_n, F_te_n, y_tr_n, y_te_n = train_test_split(F, y, test_size=0.3, random_state=0, stratify=y)
    clf_naive = models.build_baseline("rf", seed=0)
    evaluate_model(clf_naive, F_tr_n, y_tr_n, F_te_n, y_te_n, "Naive")
    
    # --- Honest Split (Recording-level, No Leakage) ---
    print("\n--- GROUPED SPLIT (Recording-level) ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(gss.split(F, y, groups))
    F_tr_g, y_tr_g = F[train_idx], y[train_idx]
    F_te_g, y_te_g = F[test_idx], y[test_idx]
    
    clf_group = models.build_baseline("rf", seed=0)
    preds_g, scores_g = evaluate_model(clf_group, F_tr_g, y_tr_g, F_te_g, y_te_g, "Grouped")
    
    # Plot ROC and Confusion for Grouped
    evaluate.plot_roc(y_te_g, scores_g, title="RF Detection ROC (Grouped)", 
                      path=os.path.join(RESULTS_DIR, "roc.png"))
    evaluate.plot_confusion(y_te_g, preds_g, labels=["Background", "Drone"], 
                            path=os.path.join(RESULTS_DIR, "confusion.png"))
                            
    print(f"Saved figures to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
