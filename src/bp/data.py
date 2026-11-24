"""Data loading, the fixed fold object and leakage-free preprocessing.

Follows the NB11 capstone harness: one seed, one StratifiedKFold object
created once and reused by every model, labels in {-1, +1}, and a
preprocessor *factory* that can only be fitted inside a fold. NB11 has no PCA
step; we add it inside the factory so PCA, like the scalers, sees training
data only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import N_FOLDS, SEED

ROOT = Path(__file__).resolve().parents[2]
FOLDS_PATH = ROOT / "data" / "folds.json"


def results_dir(week: int) -> Path:
    """week<k>/results/: the CSV and summary outputs of that week's experiments."""
    d = ROOT / f"week{week}" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(week: int) -> Path:
    """week<k>/figures/: the PDF/PNG figures made from that week's results."""
    d = ROOT / f"week{week}" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_data():
    """Breast Cancer Wisconsin (Diagnostic). y = +1 malignant, -1 benign.

    sklearn encodes malignant as 0; we make the (minority, clinically
    relevant) malignant class the positive class.
    """
    ds = load_breast_cancer()
    X = ds.data.astype(float)
    y = np.where(ds.target == 0, 1, -1)
    return X, y


def make_folds(y, n_folds: int = N_FOLDS, seed: int = SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


def save_folds(folds, y, path: Path = FOLDS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        dataset="sklearn.datasets.load_breast_cancer",
        positive_class="malignant (+1)",
        n_samples=int(len(y)),
        class_counts={"+1": int((y == 1).sum()), "-1": int((y == -1).sum())},
        seed=SEED, n_folds=len(folds), sklearn_version=sklearn.__version__,
        folds=[{"train": tr.tolist(), "test": te.tolist()} for tr, te in folds],
    )
    path.write_text(json.dumps(payload, indent=1))


def load_folds(path: Path = FOLDS_PATH):
    """The committed fold file. Every experiment reads folds from here."""
    payload = json.loads(path.read_text())
    return [(np.array(f["train"]), np.array(f["test"])) for f in payload["folds"]]


def make_preprocessor(n_components: int, bandwidth: float = 1.0, clip: bool = True):
    """Factory: ``fit(Xtr, ytr) -> transform``. Everything is learned from TRAIN only.

    StandardScaler -> PCA(n_components) -> MinMax to [0, pi] (x bandwidth).
    Test points outside the training range are clipped to [0, pi] so every
    angle stays on the encoding's monotone branch.
    """
    def fit(Xtr, ytr=None):
        std = StandardScaler().fit(Xtr)
        pca = PCA(n_components=n_components, random_state=SEED).fit(std.transform(Xtr))
        mm = MinMaxScaler((0, np.pi)).fit(pca.transform(std.transform(Xtr)))

        def transform(X):
            z = mm.transform(pca.transform(std.transform(X)))
            if clip:
                z = np.clip(z, 0, np.pi)
            return z * bandwidth

        transform.explained_variance = float(pca.explained_variance_ratio_.sum())
        return transform
    return fit
