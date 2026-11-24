"""Identity-block circuit and initializations, the teacher task, the shot
gradient of the task, and the fake-backend readout helper."""
import importlib.util

import numpy as np
import pytest

from bp.ansatz import (global_obs, hea, ib_hea, init_identity_blocks, local_obs, make_init,
                       structurally_zero_params, trivial_params)
from bp.data import ROOT
from bp.sim import zero_state
from bp.state_task import inverse_run, shot_gradient, teacher_state, train_state

rng = np.random.default_rng(2026)


def _random_state(n, b=1):
    psi = rng.normal(size=(b, 2 ** n)) + 1j * rng.normal(size=(b, 2 ** n))
    return psi / np.linalg.norm(psi, axis=1, keepdims=True)


@pytest.mark.parametrize("n,D,blocks", [(2, 1, 1), (4, 2, 1), (5, 3, 1), (4, 2, 2)])
def test_identity_blocks_are_exactly_the_identity(n, D, blocks):
    circ = ib_hea(n, D, blocks)
    assert circ.n_params == 4 * n * D * blocks
    theta = init_identity_blocks(rng, (3, circ.n_params), n, D, blocks)
    psi = _random_state(n, 3)
    np.testing.assert_allclose(circ.run(theta, psi), psi, atol=1e-12)


def test_ib_hea_keeps_its_cz_layers_and_entangles():
    """The CZ layers of the two halves must not cancel at generic angles
    (a mirrored-layer construction would make the circuit a product circuit)."""
    n, D = 4, 2
    circ = ib_hea(n, D)
    assert sum(op[0] == "cz" for op in circ.ops) == 2 * (D - 1) * (n - 1)
    psi = circ.run(rng.uniform(0, 2 * np.pi, circ.n_params), zero_state(n))[0]
    s = np.linalg.svd(psi.reshape(4, 4), compute_uv=False)   # qubits (0,1) | (2,3)
    assert (s > 1e-6).sum() > 1


def test_trivial_params_generalizes_structural_zeros():
    for n, L in [(3, 1), (4, 2), (5, 3)]:
        np.testing.assert_array_equal(trivial_params(hea(n, L)), structurally_zero_params(n, L))
    assert len(trivial_params(ib_hea(4, 2))) == 0     # ib_hea ends with R_Y


def test_reduced_domain_width():
    th = make_init("reduced", rng, (1000, 32), 4, 2)    # L = 2D = 4 -> a = pi/2
    assert np.pi / 2 >= np.abs(th).max() > 0.95 * np.pi / 2


def test_teacher_task_is_solvable_by_the_teacher_angles():
    n, D = 4, 2
    circ = ib_hea(n, D)
    theta_star = rng.uniform(0, 2 * np.pi, circ.n_params)
    psi0 = inverse_run(circ, theta_star, zero_state(n))
    assert circ.probabilities(theta_star, psi0)[0, 0] == pytest.approx(1.0)
    th_id = init_identity_blocks(rng, circ.n_params, n, D)      # identity does not solve it
    assert circ.probabilities(th_id, psi0)[0, 0] < 0.9
    assert teacher_state(circ, np.random.default_rng(0), 3).shape == (3, 2 ** n)


@pytest.mark.parametrize("obs_fn", [global_obs, local_obs])
def test_task_shot_gradient_is_unbiased(obs_fn):
    n, D = 3, 1
    circ = ib_hea(n, D)
    psi0 = teacher_state(circ, np.random.default_rng(1))
    theta = rng.uniform(0, 2 * np.pi, circ.n_params)
    obs = obs_fn(n)
    exact = -circ.expval_and_grad(theta, psi0, obs)[1][0]
    srng = np.random.default_rng(3)
    reps = np.array([shot_gradient(circ, theta, psi0, 100, srng, obs, chunk=5) for _ in range(1500)])
    se = reps.std(axis=0, ddof=1) / np.sqrt(len(reps))
    assert np.all(np.abs(reps.mean(axis=0) - exact) < 5 * se + 1e-12)


def test_train_state_is_paired_across_cells():
    h_g = train_state(4, 2, "global", "identity", seed=0, iters=2)
    h_l = train_state(4, 2, "local", "identity", seed=0, mode="shots", shots=64, iters=2)
    assert len(h_g) == 3
    # same seed -> same target and same initial angles in every cell
    assert h_g[0]["C_G"] == pytest.approx(h_l[0]["C_G"])
    assert h_g[0]["C_L"] == pytest.approx(h_l[0]["C_L"])


def test_readout_matrices_apply_per_qubit():
    spec = importlib.util.spec_from_file_location("e11", ROOT / "week2" / "experiments" / "11_fake_backend.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    A0 = np.array([[0.9, 0.2], [0.1, 0.8]])
    p = np.zeros((1, 4))
    p[0, 0] = 1.0                                     # |00>, qubit 0 = most significant bit
    np.testing.assert_allclose(m.apply_readout(p, [A0, np.eye(2)])[0], [0.9, 0, 0.1, 0])
