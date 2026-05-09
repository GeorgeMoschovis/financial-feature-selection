import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data(csv_path="american_bankruptcy_with_features.csv"):
    data = pd.read_csv(csv_path)

    target = "status_label"
    raw_cols = [c for c in data.columns if c.startswith("X") and c[1:].isdigit()]
    drop_cols = [target, "company_name", "year"] + raw_cols
    X = data.drop(columns=drop_cols)
    y = data[target].values

    X = X.fillna(X.median(numeric_only=True))
    feature_names = X.columns.tolist()
    X = X.to_numpy(dtype=np.float64)

    rng = np.random.RandomState(42)
    failed_idx = np.where(y == "failed")[0]
    alive_idx = np.where(y == "alive")[0]
    per_class = min(len(failed_idx), 1000)
    failed_keep = rng.choice(failed_idx, size=per_class, replace=False)
    alive_keep = rng.choice(alive_idx, size=per_class, replace=False)
    keep = np.concatenate([failed_keep, alive_keep])
    rng.shuffle(keep)

    X_bal = X[keep]
    y_bal = (y[keep] == "failed").astype(np.int64)
    return X_bal, y_bal, feature_names


def _print_report(selected_mask, feature_names, X, y, elapsed):
    selected_idx = np.where(selected_mask)[0]
    n_total = len(feature_names)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_preds, all_true, all_probs = [], [], []

    Xs = X[:, selected_idx]
    for tr, te in cv.split(Xs, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xs[tr])
        Xte = scaler.transform(Xs[te])
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(Xtr, y[tr])
        all_preds.extend(clf.predict(Xte))
        all_probs.extend(clf.predict_proba(Xte)[:, 1])
        all_true.extend(y[te])

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    oca = accuracy_score(all_true, all_preds)
    auc = roc_auc_score(all_true, all_probs)
    acc_alive = accuracy_score(all_true[all_true == 0], all_preds[all_true == 0])
    acc_failed = accuracy_score(all_true[all_true == 1], all_preds[all_true == 1])

    print(f"Best OCA  : {oca:.4f} ({oca*100:.2f}%)")
    print(f"AUC       : {auc:.4f}")
    print(f"Per-class : class 0 (alive) = {acc_alive:.4f}  |  class 1 (failed) = {acc_failed:.4f}")
    print(f"Features selected ({len(selected_idx)}/{n_total}):")
    for i in selected_idx:
        print(f"  [{i:2d}] {feature_names[i]}")
    print(f"Elapsed   : {elapsed:.1f}s")


def run_rfe(csv_path="american_bankruptcy_with_features.csv"):
    X, y, feature_names = load_data(csv_path)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print("\n")
    print("RFE with Random Forest (n_estimators=100)")
    print("\n")
    t0 = time.time()
    rf_est = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rfecv_rf = RFECV(estimator=rf_est, step=1, cv=cv, scoring="accuracy", n_jobs=-1)
    rfecv_rf.fit(X, y)
    elapsed = time.time() - t0
    _print_report(rfecv_rf.support_, feature_names, X, y, elapsed)


if __name__ == "__main__":
    run_rfe()
