"""The simulator is the measuring instrument; these tests calibrate it."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from bp.ansatz import global_obs, hea, local_obs, product_rx, structurally_zero_params
from bp.qiskit_backend import hea_circuit
from bp.sim import angle_encode, zero_state
from bp.variance import analytic_product_var

rng = np.random.default_rng(1234)


@pytest.mark.parametrize("n,L", [(2, 1), (3, 2), (5, 3)])
def test_matches_qiskit_statevector(n, L):
    x = rng.uniform(0, np.pi, n)
    theta = rng.uniform(0, 2 * np.pi, 2 * n * L)
    ours = hea(n, L).probabilities(theta, angle_encode(x[None]))[0]
    qc, xp, tp = hea_circuit(n, L, measure=False)
    bound = qc.assign_parameters({**dict(zip(xp, x)), **dict(zip(tp, theta))})
    theirs = Statevector(bound).reverse_qargs().probabilities()   # -> qubit 0 = MSB
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


@pytest.mark.parametrize("obs_fn", [global_obs, local_obs])
def test_adjoint_gradient_matches_finite_differences_and_parameter_shift(obs_fn):
    n, L = 4, 2
    circ, obs = hea(n, L), obs_fn(4)
    theta = rng.uniform(0, 2 * np.pi, circ.n_params)
    psi0 = angle_encode(rng.uniform(0, np.pi, (3, n)))
    f, g = circ.expval_and_grad(theta, psi0, obs)
    for k in range(circ.n_params):
        e = np.zeros_like(theta)
        e[k] = 1e-6
        fd = (circ.probabilities(theta + e, psi0) @ obs
              - circ.probabilities(theta - e, psi0) @ obs) / 2e-6
        e[k] = np.pi / 2
        ps = (circ.probabilities(theta + e, psi0) @ obs
              - circ.probabilities(theta - e, psi0) @ obs) / 2
        np.testing.assert_allclose(g[:, k], fd, atol=1e-7)
        np.testing.assert_allclose(g[:, k], ps, atol=1e-12)


def test_weighted_batch_gradient_and_per_sample_theta():
    n, L = 3, 2
    circ, obs = hea(n, L), local_obs(n)
    thetas = rng.uniform(0, 2 * np.pi, (4, circ.n_params))
    w = rng.normal(size=4)
    _, g_batch = circ.expval_and_grad(thetas, zero_state(n, 4), obs, weights=w)
    for b in range(4):
        _, g1 = circ.expval_and_grad(thetas[b], zero_state(n, 1), obs)
        np.testing.assert_allclose(g_batch[b], w[b] * g1[0], atol=1e-12)


def test_last_layer_rz_gradients_are_structurally_zero():
    n, L = 4, 2
    circ = hea(n, L)
    _, g = circ.expval_and_grad(rng.uniform(0, 2 * np.pi, (5, circ.n_params)),
                                zero_state(n, 5), global_obs(n))
    assert np.abs(g[:, structurally_zero_params(n, L)]).max() < 1e-12


def test_identity_at_zero_parameters():
    n, L = 5, 3
    circ = hea(n, L)
    p = circ.probabilities(np.zeros(circ.n_params), zero_state(n))[0]
    assert p[0] == pytest.approx(1.0)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_product_ansatz_closed_form_variances(n):
    """Var[dC_G/dtheta_1] = (1/8)(3/8)^(n-1),  Var[dC_L/dtheta_1] = 1/(8 n^2).

    Tolerance = 4 standard errors of the variance estimator. The global-cost
    gradient is a product of n-1 random factors, so it is heavy-tailed and its
    sample variance converges slowly at large n -- a fixed relative tolerance
    would be wrong in both directions.
    """
    circ = product_rx(n)
    m = 40_000
    theta = np.random.default_rng(n).uniform(0, 2 * np.pi, (m, n))
    for cost, obs in [("global", global_obs(n)), ("local", local_obs(n))]:
        _, g = circ.expval_and_grad(theta, zero_state(n, m), obs)
        x = g[:, 0] - g[:, 0].mean()
        est = x.var(ddof=1)
        se = np.sqrt((np.mean(x ** 4) - est ** 2) / m)
        assert abs(est - analytic_product_var(n, cost)) < 4 * se, cost
