"""Type ID Demo: Which drone is it? (Multi-class).

Loads DroneRF subset, extracts features, and runs Random Forest.
Uses grouped (recording-level) splitting.
"""
import os
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from spectrahawk import data_io, features, models, evaluate
from spectrahawk.synth_rf import make_synthetic_dataset

DATA_DIR = os.path.join(_REPO, "data", "dronerf")
RESULTS_DIR = os.path.join(_REPO, "results")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0 and 'README.md' not in os.listdir(DATA_DIR)[0]:
        print("Loading real DroneRF subset...")
        X, y_bin, y, groups, fs = data_io.load_dronerf(DATA_DIR, window_samples=200000)
        # Filter out background (y_bin == 0) and Phantom (y == 2)
        # AR is 1, Bebop is 0
        mask = (y_bin == 1) & (y <= 1)
        X = [X[i] for i in range(len(X)) if mask[i]]
        y = y[mask]
        groups = groups[mask]
        classes = ["Bebop", "AR"]
    else:
        print("Real dataset not found. Generating synthetic RF data (pipeline sanity check - over-separable by design)...")
        X, y, groups, fs = make_synthetic_dataset(n_per_class=100, fs=20e6, n_classes=3, seed=42)
        # Filter out background (y == 0)
        mask = (y > 0)
        X = [X[i] for i in range(len(X)) if mask[i]]
        y = y[mask]
        groups = groups[mask]
        classes = ["Type 1", "Type 2", "Type 3"]
        y = y - 1
        
    print(f"Data: {len(X)} drone segments.")
    print(f"Recordings (groups): {len(np.unique(groups))}")
    
    print("\n[CAVEAT] With very few independent recordings, grouped type-ID is statistically thin.")
    print("Results are indicative of pipeline functioning, not definitive real-world performance.\n")
    
    print("Extracting features...")
    F = models.features_matrix(X, fs, features.feature_vector)
    
    # --- NAIVE SPLIT (Segment-level) ---
    from sklearn.model_selection import train_test_split
    F_tr_n, F_te_n, y_tr_n, y_te_n = train_test_split(F, y, test_size=0.3, random_state=0, stratify=y)
    
    print("\n--- NAIVE SPLIT (File/Clip-level) ---")
    clf_naive = models.build_baseline("rf", seed=0)
    clf_naive.fit(F_tr_n, y_tr_n)
    preds_n = clf_naive.predict(F_te_n)
    print(f"[Naive] Accuracy : {accuracy_score(y_te_n, preds_n):.4f}")
    print(f"[Naive] Macro-F1 : {f1_score(y_te_n, preds_n, average='macro'):.4f}")
    print("------------------------------\n")

    # --- GROUPED SPLIT (Recording-level) ---
    print("--- GROUPED SPLIT (Recording-level) ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(gss.split(F, y, groups))
    F_train, y_train = F[train_idx], y[train_idx]
    F_test, y_test = F[test_idx], y[test_idx]
    
    clf = models.build_baseline("rf", seed=0)
    clf.fit(F_train, y_train)
    
    preds = clf.predict(F_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')
    
    print(f"[Grouped] Accuracy : {acc:.4f}")
    print(f"[Grouped] Macro-F1 : {macro_f1:.4f}\n")
    
    labels = np.unique(np.concatenate((y_test, preds)))
    target_names = [classes[int(i)] for i in labels]
    
    print("Grouped Classification Report:")
    print(classification_report(y_test, preds, labels=labels, target_names=target_names))
    
    # Plot Confusion Matrix
    evaluate.plot_confusion(y_test, preds, labels=target_names, 
                            path=os.path.join(RESULTS_DIR, "typeid_confusion.png"))
                            
    print(f"Saved figure to {RESULTS_DIR}/typeid_confusion.png")

if __name__ == "__main__":
    main()
