from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

EPS = 1e-7


def bce(p, t):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def scores(y, p, threshold=0.5):
    """y in {-1,+1}; p = model probability of class +1."""
    pred = np.where(p >= threshold, 1, -1)
    t = (y == 1).astype(float)
    return dict(acc=accuracy_score(y, pred),
                bacc=balanced_accuracy_score(y, pred),
                auc=roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
                loss=bce(p, t))
