"""Classical baselines on exactly the same folds and preprocessed features."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from . import SEED
from .metrics import bce


def mlp_params(n: int, h: int) -> int:
    """Weights + biases of an n-h-1 MLP with one logistic output unit."""
    return n * h + h + h + 1


def matched_hidden(n: int, target: int) -> int:
    """Hidden width whose parameter count is closest to ``target`` (ties -> larger)."""
    cands = range(1, 64)
    return min(cands, key=lambda h: (abs(mlp_params(n, h) - target), -h))


def _metrics(y, pred, score):
    return dict(acc=accuracy_score(y, pred), bacc=balanced_accuracy_score(y, pred),
                auc=roc_auc_score(y, score) if len(np.unique(score)) > 1 else 0.5)


def run_baseline(name, Xtr, ytr, Xte, yte, **kw):
    if name == "majority":
        m = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
        return _metrics(yte, m.predict(Xte), np.zeros(len(Xte)))
    if name == "logreg":
        m = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
        return _metrics(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])
    if name == "svm_rbf":
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED + 1)
        m = GridSearchCV(SVC(), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 1]},
                         cv=inner).fit(Xtr, ytr)
        return _metrics(yte, m.predict(Xte), m.decision_function(Xte))
    if name == "mlp":
        m, _ = mlp_with_curves(Xtr, ytr, Xte, yte, **kw)
        return _metrics(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])
    raise ValueError(name)


def mlp_with_curves(Xtr, ytr, Xte, yte, hidden, epochs=60, lr=0.01,
                    batch_size=32, seed=SEED):
    """Parameter-matched MLP trained epoch by epoch, logging the same curves as
    the quantum model (same epochs, batch size, loss and metrics)."""
    m = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh", solver="adam",
                      learning_rate_init=lr, batch_size=batch_size, random_state=seed)
    ttr, tte = (ytr == 1).astype(float), (yte == 1).astype(float)
    history = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, epochs + 1):
            m.partial_fit(Xtr, ytr, classes=np.array([-1, 1]))   # one pass = one epoch
            ptr, pte = m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]
            row = dict(epoch=epoch, train_loss=bce(ptr, ttr), test_loss=bce(pte, tte))
            row.update({f"test_{k}": v for k, v in
                        _metrics(yte, m.predict(Xte), pte).items()})
            history.append(row)
    return m, history
