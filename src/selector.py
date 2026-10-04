"""Learned scheduler selector.

For each scheduler s we train a small binary model estimating
    P(s achieves the minimum total seek on this window | window features).
The selector picks the scheduler with the highest probability; that probability
is its CONFIDENCE, so "confidence" literally means "chance my pick is optimal".
Ties in seek are therefore never counted as mistakes. If two schedulers have
exactly the same probability, the tie-break prefers the more starvation-resistant
one (labels.TIE_PREFERENCE).
"""
from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from schedulers import NAMES
from labels import TIE_PREFERENCE


def make_model(kind: str):
    if kind == "tree":     # main model: shallow, readable decision tree
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0)
    if kind == "logreg":   # comparison model
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    raise ValueError(kind)


class SchedulerSelector:
    def __init__(self, kind: str = "tree"):
        self.kind = kind
        self.models = {}

    def fit(self, X, optimal_sets):
        X = np.asarray(X)
        for s in NAMES:
            y = np.array([s in o for o in optimal_sets], dtype=int)
            if y.min() == y.max():            # scheduler always / never optimal in training
                self.models[s] = float(y[0])
            else:
                self.models[s] = make_model(self.kind).fit(X, y)
        return self

    def proba(self, X):
        X = np.atleast_2d(np.asarray(X))
        cols = [np.full(len(X), m) if isinstance(m, float) else m.predict_proba(X)[:, 1]
                for m in (self.models[s] for s in NAMES)]
        return np.column_stack(cols)

    def select(self, x):
        p = self.proba(x)[0]
        best = p.max()
        pick = next(s for s in TIE_PREFERENCE if p[NAMES.index(s)] == best)
        return pick, float(best), p


def calibration_stats(conf, correct, n_bins: int = 10):
    """ECE (Guo et al., 2017), Brier score (Brier, 1950) and a reliability table."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - cf)
        table.append((lo, hi, int(m.sum()), cf, acc))
    brier = float(np.mean((conf - correct) ** 2))
    return {"ECE": float(ece), "Brier": brier,
            "mean_conf_correct": float(conf[correct == 1].mean()) if (correct == 1).any() else np.nan,
            "mean_conf_wrong": float(conf[correct == 0].mean()) if (correct == 0).any() else np.nan,
            "accuracy": float(correct.mean())}, table
