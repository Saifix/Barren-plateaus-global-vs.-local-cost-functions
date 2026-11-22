"""The density-matrix simulator (noise study) against Aer, finite differences
and the state-vector simulator."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error

from bp.ansatz import global_obs, hea, ib_hea, local_obs
from bp.density import DensityCircuit, Noise, density_from_states
from bp.sim import angle_encode, zero_state

rng = np.random.default_rng(99)
NOISES = [Noise(p1=0.03), Noise(g1=0.05), Noise(p2=0.08), Noise(g2=0.1),
          Noise(p1=0.03, p2=0.08, g1=0.05, g2=0.1)]


def test_noiseless_matches_statevector():
    n, L = 3, 2
    circ = hea(n, L)
    th = rng.uniform(0, 2 * np.pi, (2, circ.n_params))
    psi = angle_encode(rng.uniform(0, np.pi, (2, n)))
    dc = DensityCircuit(circ)
    np.testing.assert_allclose(dc.probabilities(th, density_from_states(psi)),
                               circ.probabilities(th, psi), atol=1e-12)
    for obs in (global_obs(n), local_obs(n)):
        _, g1 = dc.expval_and_grad(th, density_from_states(psi), obs)
        _, g2 = circ.expval_and_grad(th, psi, obs)
        np.testing.assert_allclose(g1, g2, atol=1e-12)


@pytest.mark.parametrize("noise", NOISES)
@pytest.mark.parametrize("make", [hea, ib_hea])
def test_matches_aer_density_matrix(noise, make):
    n = 3
    circ = make(n, 2)
    th = rng.uniform(0, 2 * np.pi, circ.n_params)
    qc = QuantumCircuit(n)
    for kind, q, idx in circ.ops:
        if kind == "cz":
            qc.cz(*q)
        else:
            getattr(qc, kind)(th[idx], q)
    qc.save_density_matrix()
    nm = NoiseModel(basis_gates=["ry", "rz", "cz"])
    ad2 = amplitude_damping_error(noise.g2)
    nm.add_all_qubit_quantum_error(
        depolarizing_error(noise.p1, 1).compose(amplitude_damping_error(noise.g1)), ["ry", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(noise.p2, 2).compose(ad2.tensor(ad2)), ["cz"])
    res = AerSimulator(method="density_matrix", noise_model=nm).run(qc).result()
    theirs = DensityMatrix(res.data()["density_matrix"]).reverse_qargs().data
    ours = DensityCircuit(circ, noise).run(th, density_from_states(zero_state(n)))[0]
    np.testing.assert_allclose(ours.reshape(2 ** n, 2 ** n), theirs, atol=1e-12)


@pytest.mark.parametrize("noise", NOISES)
def test_noisy_adjoint_gradient_matches_finite_differences(noise):
    n, L = 3, 2
    circ = hea(n, L)
    dc = DensityCircuit(circ, noise)
    th = rng.uniform(0, 2 * np.pi, circ.n_params)
    rho0 = density_from_states(angle_encode(rng.uniform(0, np.pi, (1, n))))
    for obs in (global_obs(n), local_obs(n)):
        _, g = dc.expval_and_grad(th, rho0, obs)
        for k in range(circ.n_params):
            e = np.zeros_like(th)
            e[k] = 1e-6
            fd = (dc.probabilities(th + e, rho0) @ obs - dc.probabilities(th - e, rho0) @ obs) / 2e-6
            assert g[0, k] == pytest.approx(fd[0], abs=1e-8)


def test_trace_preserved_and_damping_fixes_zero_state():
    n = 3
    circ = hea(n, 2)
    th = rng.uniform(0, 2 * np.pi, circ.n_params)
    p = DensityCircuit(circ, NOISES[-1]).probabilities(th, density_from_states(zero_state(n)))
    assert p.sum() == pytest.approx(1.0)
    # pure amplitude damping with all angles 0 leaves |0...0> unchanged
    p0 = DensityCircuit(circ, Noise(g1=0.3, g2=0.3)).probabilities(
        np.zeros(circ.n_params), density_from_states(zero_state(n)))
    assert p0[0, 0] == pytest.approx(1.0)
