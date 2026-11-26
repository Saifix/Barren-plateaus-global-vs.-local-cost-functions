"""The Pauli-path model gives the exact gradient variance over uniform random angles."""
import numpy as np
import pytest

from bp.ansatz import hea, init_uniform, observable, structurally_zero_params
from bp.density import DensityCircuit, Noise, density_from_states
from bp.pauli_path import PathNoise, gradient_variances, mean_pauli_weight
from bp.sim import zero_state


@pytest.mark.parametrize("cost", ["global", "local"])
def test_matches_monte_carlo_noiseless(cost):
    n, L, draws = 3, 2, 30000
    circ = hea(n, L)
    theta = init_uniform(np.random.default_rng(5), (draws, circ.n_params))
    _, g = circ.expval_and_grad(theta, zero_state(n, draws), observable(n, cost))
    mc = g.var(axis=0, ddof=1)
    model = gradient_variances(n, L, cost)
    for k in range(circ.n_params):
        if mc[k] > 1e-10:
            assert model[k] == pytest.approx(mc[k], rel=0.06)


@pytest.mark.parametrize("cost", ["global", "local"])
def test_matches_density_simulation_depolarizing(cost):
    n, L, draws, p2 = 2, 2, 8000, 0.1
    circ = hea(n, L)
    theta = init_uniform(np.random.default_rng(6), (draws, circ.n_params))
    dc = DensityCircuit(circ, Noise(p1=p2 / 10, p2=p2))
    _, g = dc.expval_and_grad(theta, density_from_states(zero_state(n, draws)), observable(n, cost))
    mc = g.var(axis=0, ddof=1)
    model = gradient_variances(n, L, cost, PathNoise(p1=p2 / 10, p2=p2))
    for k in range(circ.n_params):
        if mc[k] > 1e-10:
            assert model[k] == pytest.approx(mc[k], rel=0.08)


def test_structural_zeros_and_noise_monotone():
    n, L = 4, 2
    v0 = gradient_variances(n, L, "global")
    for k in structurally_zero_params(n, L):
        assert v0[k] == pytest.approx(0, abs=1e-15)
    v1 = gradient_variances(n, L, "global", PathNoise(p1=0.001, p2=0.01))
    assert all(v1[k] <= v0[k] + 1e-15 for k in v0)


def test_local_weight_bounded_by_light_cone():
    w = mean_pauli_weight(8, 3, "local")
    assert w[0] < 3.0 and all(a <= b + 1e-9 for a, b in zip(w, w[1:]))
