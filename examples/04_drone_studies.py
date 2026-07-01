import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from spectrahawk import data_io, features, models  # noqa: E402
from spectrahawk.splits import drone_side_grouped_detection_splits  # noqa: E402

DATA_DIR = os.path.join(_REPO, "data", "dronerf")
TYPE_ID_LABELS = [0, 1]  # Bebop, AR


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    return np.percentile(vals, [2.5, 97.5])


def pooled_scores(y_true, y_pred, labels=TYPE_ID_LABELS, n_boot=1000, seed=0):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    f1_ci = bootstrap_ci(
        y_true,
        y_pred,
        lambda yt, yp: f1_score(yt, yp, labels=labels, average="macro", zero_division=0),
        n_boot=n_boot,
        seed=seed,
    )
    acc_ci = bootstrap_ci(y_true, y_pred, accuracy_score, n_boot=n_boot, seed=seed + 1)
    return macro_f1, f1_ci, acc, acc_ci


def format_metric(mean, ci):
    return f"{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def custom_detection_loro(y_bin, groups, n_splits=9, seed=0):
    """Drone-side LORO with segment-split background.

    Drone recordings are left out as groups. Background is split at the segment
    level because DroneRF has only one independent background recording, so this
    protocol has residual background leakage.
    """
    yield from drone_side_grouped_detection_splits(y_bin, groups, n_splits=n_splits, seed=seed)


def get_pd_at_fa(y_true, y_scores, fa_target=0.01):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.where(fpr <= fa_target)[0][-1]
    return tpr[idx]


def study_2_type_id_loro(F, y, groups):
    print("\n--- STUDY 2: Type-ID LORO (AR vs Bebop) ---")

    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=0)
    naive_oof = np.empty_like(y)
    naive_seen = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(F, y):
        clf = models.build_baseline("rf", seed=0)
        clf.fit(F[tr], y[tr])
        naive_oof[te] = clf.predict(F[te])
        naive_seen[te] += 1
    if not np.all(naive_seen == 1):
        raise AssertionError("Naive OOF predictions do not cover every segment exactly once.")

    logo = LeaveOneGroupOut()
    grouped_oof = np.empty_like(y)
    grouped_seen = np.zeros(len(y), dtype=int)
    for tr, te in logo.split(F, y, groups):
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("LORO recording groups overlap between train and test.")
        clf = models.build_baseline("rf", seed=0)
        clf.fit(F[tr], y[tr])
        grouped_oof[te] = clf.predict(F[te])
        grouped_seen[te] += 1
    if not np.all(grouped_seen == 1):
        raise AssertionError("Grouped LORO OOF predictions do not cover every segment exactly once.")

    naive_f1, naive_f1_ci, naive_acc, naive_acc_ci = pooled_scores(y, naive_oof, seed=0)
    grouped_f1, grouped_f1_ci, grouped_acc, grouped_acc_ci = pooled_scores(y, grouped_oof, seed=1)

    print(
        "Naive pooled Macro-F1: "
        f"{format_metric(naive_f1, naive_f1_ci)} | Accuracy: {format_metric(naive_acc, naive_acc_ci)}"
    )
    print(
        "Grouped LORO pooled Macro-F1: "
        f"{format_metric(grouped_f1, grouped_f1_ci)} | Accuracy: {format_metric(grouped_acc, grouped_acc_ci)}"
    )


def study_2_detection_loro(F, y_bin, groups):
    print("\n--- STUDY 2: Detection, drone-side grouped (residual background leakage) ---")

    drone_groups = np.unique(groups[y_bin == 1])
    n_drone_groups = len(drone_groups)

    skf = StratifiedKFold(n_splits=n_drone_groups, shuffle=True, random_state=0)
    naive_aucs, naive_pds = [], []
    for tr, te in skf.split(F, y_bin):
        clf = models.build_baseline("rf", seed=0)
        clf.fit(F[tr], y_bin[tr])
        scores = clf.predict_proba(F[te])[:, 1]
        naive_aucs.append(roc_auc_score(y_bin[te], scores))
        naive_pds.append(get_pd_at_fa(y_bin[te], scores, 0.01))

    grouped_aucs, grouped_pds = [], []
    for tr, te in custom_detection_loro(y_bin, groups, n_splits=n_drone_groups, seed=0):
        clf = models.build_baseline("rf", seed=0)
        clf.fit(F[tr], y_bin[tr])
        scores = clf.predict_proba(F[te])[:, 1]
        grouped_aucs.append(roc_auc_score(y_bin[te], scores))
        grouped_pds.append(get_pd_at_fa(y_bin[te], scores, 0.01))

    print(
        f"Naive ROC-AUC: {np.mean(naive_aucs):.4f} +/- {np.std(naive_aucs):.4f} | "
        f"Pd@1%FA: {np.mean(naive_pds):.4f} +/- {np.std(naive_pds):.4f}"
    )
    print(
        "Drone-side grouped (residual background leakage) ROC-AUC: "
        f"{np.mean(grouped_aucs):.4f} +/- {np.std(grouped_aucs):.4f} | "
        f"Pd@1%FA: {np.mean(grouped_pds):.4f} +/- {np.std(grouped_pds):.4f}"
    )


def study_3_ablation(F, y, groups_l1, groups_l2):
    print("\n--- STUDY 3: Leakage-pathway Ablation (AR vs Bebop Type-ID) ---")

    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    l0_oof = np.empty_like(y)
    l0_seen = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(F, y):
        clf = models.build_baseline("rf", seed=0)
        clf.fit(F[tr], y[tr])
        l0_oof[te] = clf.predict(F[te])
        l0_seen[te] += 1

    gkf_l1 = GroupKFold(n_splits=n_splits)
    l1_oof = np.empty_like(y)
    l1_seen = np.zeros(len(y), dtype=int)
    for tr, te in gkf_l1.split(F, y, groups_l1):
        if set(groups_l1[tr]) & set(groups_l1[te]):
            raise AssertionError("L1 groups overlap between train and test.")
        clf = models.build_baseline("rf", seed=0)
        clf.fit(F[tr], y[tr])
        l1_oof[te] = clf.predict(F[te])
        l1_seen[te] += 1

    gkf_l2 = GroupKFold(n_splits=n_splits)
    l2_oof = np.empty_like(y)
    l2_seen = np.zeros(len(y), dtype=int)
    for tr, te in gkf_l2.split(F, y, groups_l2):
        if set(groups_l2[tr]) & set(groups_l2[te]):
            raise AssertionError("L2 recording groups overlap between train and test.")
        clf = models.build_baseline("rf", seed=0)
        clf.fit(F[tr], y[tr])
        l2_oof[te] = clf.predict(F[te])
        l2_seen[te] += 1
    if not (np.all(l0_seen == 1) and np.all(l1_seen == 1) and np.all(l2_seen == 1)):
        raise AssertionError("Ablation OOF predictions do not cover every segment exactly once.")

    l0_f1, _, l0_acc, _ = pooled_scores(y, l0_oof, seed=10)
    l1_f1, _, l1_acc, _ = pooled_scores(y, l1_oof, seed=11)
    l2_f1, _, l2_acc, _ = pooled_scores(y, l2_oof, seed=12)

    print(f"L0 (Segment-level) Macro-F1: {l0_f1:.4f} | Accuracy: {l0_acc:.4f}")
    print(f"L1 (File-level)    Macro-F1: {l1_f1:.4f} | Accuracy: {l1_acc:.4f}")
    print(f"L2 (Record-level)  Macro-F1: {l2_f1:.4f} | Accuracy: {l2_acc:.4f}")


def main():
    print("Loading DroneRF features for Studies 2 & 3...")
    F, y_bin, y, groups_l2, _, groups_l1 = data_io.load_dronerf_features(
        DATA_DIR,
        features.feature_vector,
        window_samples=200000,
        batch_extract_fn=features.feature_matrix_windows,
    )

    study_2_detection_loro(F, y_bin, groups_l2)

    mask_tid = (y_bin == 1) & (y <= 1)
    F_tid = F[mask_tid]
    y_tid = y[mask_tid]
    groups_l2_tid = groups_l2[mask_tid]
    groups_l1_tid = groups_l1[mask_tid]

    study_2_type_id_loro(F_tid, y_tid, groups_l2_tid)
    study_3_ablation(F_tid, y_tid, groups_l1_tid, groups_l2_tid)


if __name__ == "__main__":
    main()
