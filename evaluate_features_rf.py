import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_data(csv_path="american_bankruptcy_with_features.csv", seed=42, sample_size=2000):
    data = pd.read_csv(csv_path)

    target = "status_label"
    raw_cols = [c for c in data.columns if c.startswith("X") and c[1:].isdigit()]
    drop_cols = [target, "company_name", "year"] + raw_cols
    X = data.drop(columns=drop_cols)
    y = data[target]

    X = X.fillna(X.median(numeric_only=True))

    rng = np.random.RandomState(seed)
    failed_idx = np.where(y == "failed")[0]
    alive_idx = np.where(y == "alive")[0]

    per_class = len(failed_idx) if sample_size == 0 else min(len(failed_idx), sample_size // 2)
    failed_keep = rng.choice(failed_idx, size=per_class, replace=False) \
                  if per_class < len(failed_idx) else failed_idx
    alive_keep = rng.choice(alive_idx, size=per_class, replace=False)
    keep = np.concatenate([failed_keep, alive_keep])
    rng.shuffle(keep)

    X_bal = X.iloc[keep].reset_index(drop=True).to_numpy(dtype=np.float64)
    y_bal = (y.iloc[keep].values == "failed").astype(np.int64)
    return X_bal, y_bal, X.columns.tolist()


class EvaluatorRF:
    """Same interface as Evaluator but uses Random Forest instead of kNN."""

    def __init__(self, csv_path="american_bankruptcy_with_features.csv", cv_folds=10, sample_size=2000, seed=42, n_estimators=20):
        self.seed = seed
        self.n_estimators = n_estimators
        self.X, self.y, self.feature_names = load_data(csv_path, seed, sample_size)
        self.num_features = self.X.shape[1]
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        self.cv_splits = list(skf.split(self.X, self.y))
        self._cache = {}

    def evaluate(self, subset: frozenset) -> float:
        mask = np.array([1 if i in subset else 0 for i in range(self.num_features)], dtype=np.int8)
        return self._score(mask)

    def evaluate_from_string(self, mask_string: str) -> float:
        if len(mask_string) != self.num_features:
            raise ValueError(f"expected {self.num_features} bits, got {len(mask_string)}")
        mask = np.array([int(c) for c in mask_string], dtype=np.int8)
        return self._score(mask)

    def _score(self, mask: np.ndarray) -> float:
        key = mask.tobytes()
        if key in self._cache:
            return self._cache[key]

        selected = np.where(mask == 1)[0]
        if selected.size == 0:
            self._cache[key] = 0.0
            return 0.0

        Xs = self.X[:, selected]
        accs = []
        for tr, te in self.cv_splits:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xs[tr])
            Xte = scaler.transform(Xs[te])
            clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.seed)
            clf.fit(Xtr, self.y[tr])
            accs.append(accuracy_score(self.y[te], clf.predict(Xte)))

        oca = float(np.mean(accs))
        self._cache[key] = oca
        return oca

    def final_report(self, subset: frozenset) -> dict:
        mask = np.array([1 if i in subset else 0 for i in range(self.num_features)], dtype=np.int8)
        selected = np.where(mask == 1)[0]

        all_preds, all_true, all_probs = [], [], []
        for tr, te in self.cv_splits:
            Xs = self.X[:, selected]
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xs[tr])
            Xte = scaler.transform(Xs[te])
            clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.seed)
            clf.fit(Xtr, self.y[tr])
            all_preds.extend(clf.predict(Xte))
            all_true.extend(self.y[te])
            all_probs.extend(clf.predict_proba(Xte)[:, 1])

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        all_probs = np.array(all_probs)

        oca = accuracy_score(all_true, all_preds)
        auc = roc_auc_score(all_true, all_probs)
        per_class = {int(c): accuracy_score(all_true[all_true == c], all_preds[all_true == c]) for c in np.unique(all_true)}
        aca = float(np.mean(list(per_class.values())))
        rmse = float(np.sqrt(np.mean((all_preds - all_true) ** 2)))

        return {"OCA": round(oca, 4), "AUC": round(auc, 4), "ACA": round(aca, 4), "RMSE": round(rmse, 4),
                "per_class_accuracy": {k: round(v, 4) for k, v in per_class.items()},
                "n_features": len(subset)}

    @property
    def cache_size(self):
        return len(self._cache)
