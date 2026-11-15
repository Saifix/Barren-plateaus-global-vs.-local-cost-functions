"""Fold hygiene and preprocessing (the parts NB11 says are easy to get silently wrong)."""
import numpy as np
from qiskit_aer import AerSimulator

from bp.classical import matched_hidden, mlp_params
from bp.data import load_data, load_folds, make_folds, make_preprocessor, FOLDS_PATH
from bp.qclassifier import QuantumClassifier
from bp.qiskit_backend import sampled_scores


def test_folds_partition_and_stratify():
    X, y = load_data()
    folds = make_folds(y)
    test_idx = np.concatenate([te for _, te in folds])
    assert sorted(test_idx) == list(range(len(y)))
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert abs((y[te] == 1).mean() - (y == 1).mean()) < 0.02


def test_committed_folds_match_regenerated():
    if not FOLDS_PATH.exists():
        return
    _, y = load_data()
    for (a_tr, a_te), (b_tr, b_te) in zip(load_folds(), make_folds(y)):
        assert np.array_equal(a_te, b_te) and np.array_equal(a_tr, b_tr)


def test_preprocessor_uses_train_only():
    X, y = load_data()
    tr, te = make_folds(y)[0]
    t1 = make_preprocessor(4)(X[tr], y[tr])
    X_poisoned = X.copy()
    X_poisoned[te] *= 1000                    # changing TEST rows must not change the fit
    t2 = make_preprocessor(4)(X_poisoned[tr], y[tr])
    np.testing.assert_allclose(t1(X[tr]), t2(X[tr]))
    z = t1(X[te])
    assert z.shape == (len(te), 4) and z.min() >= 0 and z.max() <= np.pi


def test_parameter_matching():
    assert mlp_params(4, 3) == 19
    assert matched_hidden(4, 16) == 3


def test_aer_sampled_scores_agree_with_exact():
    n, L = 3, 1
    clf = QuantumClassifier(n, L, readout="local")
    clf.theta = np.random.default_rng(0).uniform(0, 2 * np.pi, clf.n_params)
    X = np.random.default_rng(1).uniform(0, np.pi, (6, n))
    for readout in ("global", "local"):
        clf.obs = __import__("bp.ansatz", fromlist=["observable"]).observable(n, readout)
        exact = clf.predict_proba(X)
        est = sampled_scores(AerSimulator(), n, L, X, clf.theta, readout, shots=20000, seed=3)
        np.testing.assert_allclose(est, exact, atol=0.02)
