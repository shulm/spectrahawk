import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_REPO, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_data(R=8, S=50, lam=2.0, delta=0.7706, seed=0):
    rng = np.random.default_rng(seed)
    d = 20
    sigma = 1.0

    X = []
    y = []
    groups = []

    group_id = 0
    for class_label in [0, 1]:
        mu_c = np.zeros(d)
        mu_c[0] = -delta / 2 if class_label == 0 else delta / 2

        for _ in range(R):
            # Recording-specific nuisance is confined to non-discriminative axes.
            nuisance = rng.normal(0, lam, size=d)
            nuisance[0] = 0.0

            for _ in range(S):
                epsilon = rng.normal(0, sigma, size=d)
                X.append(mu_c + nuisance + epsilon)
                y.append(class_label)
                groups.append(group_id)
            group_id += 1

    return np.array(X), np.array(y), np.array(groups)


def eval_protocol(X, y, groups, n_splits=5, grouped=False, clf_type="rf"):
    if grouped:
        n_grps = len(np.unique(groups))
        actual_splits = min(n_grps // 2, n_splits)
        actual_splits = max(actual_splits, 2)
        cv = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=0)
        splits = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = cv.split(X, y)

    accs = []
    for train_idx, test_idx in splits:
        if grouped and set(groups[train_idx]) & set(groups[test_idx]):
            raise AssertionError("Grouped split leaked recording groups between train and test.")
        if clf_type == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        elif clf_type == "lr":
            clf = LogisticRegression(random_state=0, max_iter=1000)
        else:
            raise ValueError(f"Unknown classifier type: {clf_type}")

        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accs.append(balanced_accuracy_score(y[test_idx], preds))

    return float(np.mean(accs))


def mean_ci(vals):
    vals = np.asarray(vals, dtype=float)
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return mean, 0.0
    half_width = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
    return mean, half_width


def fmt(mean, ci):
    return f"{mean:.4f} +/- {ci:.4f}"


def main():
    sigma = 1.0
    delta = 0.7706  # chosen for true Bayes balanced accuracy approx 0.65
    bayes_acc = norm.cdf(delta / (2 * sigma))
    seeds = list(range(10))

    print(f"Dataset seeds: {seeds}")
    print(f"True (Bayes) balanced accuracy: {bayes_acc:.4f}")

    lambdas = [0, 0.5, 1, 2, 4, 8]
    lambda_rows = []

    print("\n--- Sweep 1: Fix R=8, vary lambda ---")
    for lam in lambdas:
        naive_rf, grouped_rf, grouped_lr = [], [], []
        for seed in seeds:
            X, y, groups = generate_data(R=8, lam=lam, delta=delta, seed=seed)
            naive_rf.append(eval_protocol(X, y, groups, n_splits=5, grouped=False, clf_type="rf"))
            grouped_rf.append(eval_protocol(X, y, groups, n_splits=5, grouped=True, clf_type="rf"))
            grouped_lr.append(eval_protocol(X, y, groups, n_splits=5, grouped=True, clf_type="lr"))

        n_mean, n_ci = mean_ci(naive_rf)
        g_mean, g_ci = mean_ci(grouped_rf)
        lr_mean, lr_ci = mean_ci(grouped_lr)
        lambda_rows.append((lam, n_mean, n_ci, g_mean, g_ci, lr_mean, lr_ci))
        print(
            f"lambda={lam:<4} | RF Naive: {fmt(n_mean, n_ci)} | "
            f"RF Grouped: {fmt(g_mean, g_ci)} | LR Grouped: {fmt(lr_mean, lr_ci)}"
        )

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        lambdas,
        [row[1] for row in lambda_rows],
        yerr=[row[2] for row in lambda_rows],
        label="RF Naive (Leakage)",
        marker="o",
    )
    plt.errorbar(
        lambdas,
        [row[3] for row in lambda_rows],
        yerr=[row[4] for row in lambda_rows],
        label="RF Grouped (Honest)",
        marker="s",
    )
    plt.axhline(bayes_acc, color="r", linestyle="--", label="True Bayes Acc")
    plt.xlabel(r"Nuisance Scale ($\lambda$)")
    plt.ylabel("Balanced Accuracy")
    plt.title("Leakage vs Nuisance Scale (R=8 recordings/class)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "leakage_vs_lambda.png"), dpi=130)
    plt.close()

    Rs = [2, 4, 8, 16, 32]
    r_rows = []

    print("\n--- Sweep 2: Fix lambda=2, vary R (recordings/class) ---")
    for r in Rs:
        naive_rf, grouped_rf = [], []
        n_splits = min(5, 2 * r)
        for seed in seeds:
            X, y, groups = generate_data(R=r, lam=2.0, delta=delta, seed=seed)
            naive_rf.append(eval_protocol(X, y, groups, n_splits=n_splits, grouped=False, clf_type="rf"))
            grouped_rf.append(eval_protocol(X, y, groups, n_splits=n_splits, grouped=True, clf_type="rf"))

        n_mean, n_ci = mean_ci(naive_rf)
        g_mean, g_ci = mean_ci(grouped_rf)
        r_rows.append((r, n_mean, n_ci, g_mean, g_ci))
        print(f"R={r:<2} | RF Naive: {fmt(n_mean, n_ci)} | RF Grouped: {fmt(g_mean, g_ci)}")

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        Rs,
        [row[1] for row in r_rows],
        yerr=[row[2] for row in r_rows],
        label="RF Naive (Leakage)",
        marker="o",
    )
    plt.errorbar(
        Rs,
        [row[3] for row in r_rows],
        yerr=[row[4] for row in r_rows],
        label="RF Grouped (Honest)",
        marker="s",
    )
    plt.axhline(bayes_acc, color="r", linestyle="--", label="True Bayes Acc")
    plt.xlabel("Recordings per Class (R)")
    plt.ylabel("Balanced Accuracy")
    plt.title(r"Leakage vs Recordings (Nuisance $\lambda=2$)")
    plt.xscale("log")
    plt.xticks(Rs, labels=[str(r) for r in Rs])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "leakage_vs_recordings.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
