"""The shot-based gradient must be an unbiased, correctly scaled estimate."""
import numpy as np
import pytest

from bp.qclassifier import QuantumClassifier


@pytest.mark.parametrize("readout", ["global", "local"])
def test_shot_gradient_is_unbiased(readout):
    n, L = 3, 2
    rng = np.random.default_rng(5)
    X = rng.uniform(0, np.pi, (6, n))
    t = np.array([1, 0, 1, 1, 0, 0], float)
    clf = QuantumClassifier(n, L, readout, loss="brier", shots=200)
    theta = rng.uniform(0, 2 * np.pi, clf.n_params)
    _, exact = clf.exact_grad(theta, X, t)
    reps = np.array([clf.shot_grad(theta, X, t, rng)[1] for _ in range(1500)])
    se = reps.std(axis=0, ddof=1) / np.sqrt(len(reps))
    assert np.all(np.abs(reps.mean(axis=0) - exact) < 5 * se + 1e-12)


def test_exact_gradient_matches_finite_difference_for_brier():
    n, L = 3, 1
    rng = np.random.default_rng(6)
    X, t = rng.uniform(0, np.pi, (5, n)), np.array([1, 0, 1, 0, 1], float)
    clf = QuantumClassifier(n, L, "local", loss="brier")
    theta = rng.uniform(0, 2 * np.pi, clf.n_params)
    _, g = clf.exact_grad(theta, X, t)
    for k in range(clf.n_params):
        e = np.zeros_like(theta); e[k] = 1e-6
        fd = (clf.exact_grad(theta + e, X, t)[0] - clf.exact_grad(theta - e, X, t)[0]) / 2e-6
        assert g[k] == pytest.approx(fd, abs=1e-7)


def test_bce_with_shots_is_refused():
    with pytest.raises(ValueError):
        QuantumClassifier(3, 1, "global", loss="bce", shots=100)
