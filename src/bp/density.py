"""Density-matrix simulator with gate noise and exact adjoint gradients.

Used for the noise study (unital depolarizing vs. non-unital amplitude damping).
It reuses the gate kernels of ``sim``: a density matrix rho on n qubits is
stored as a flattened ``(B, 4**n)`` array, which is a state vector on 2n
"qubits" (row qubit q = index q, column qubit q = index n + q). A unitary U
acts as U on the row qubit and conj(U) on the column qubit.

Noise model (``Noise``): after every single-qubit gate, a depolarizing channel
with parameter ``p1`` and/or an amplitude-damping channel with ``g1`` on that
qubit; after every CZ, a two-qubit depolarizing channel with ``p2`` and/or
amplitude damping with ``g2`` on both qubits. Parameters follow Qiskit's
conventions, so the same model can be built in Aer (``tests/`` compares them):
  depolarizing(lam):      rho -> (1 - lam) rho + lam * I/d (x) Tr_q rho
  amplitude damping(g):   Kraus [[1,0],[0,sqrt(1-g)]], [[0,sqrt(g)],[0,0]]

Gradients: f = Tr[O E_M(...E_1(rho0))] with Hermitian, Z-diagonal O. Going
backwards, Lambda_j = E_{j+1}^dag(...E_M^dag(O)) (the observable in the
Heisenberg picture). For a rotation R_P(t) = exp(-i t P/2), whose output state
is rho_j,
    d f / d t = Im Tr[Lambda_j P rho_j],
which is the density-matrix form of the state-vector adjoint rule. The
adjoint channels are contractive, so the backward pass is numerically stable
for any noise strength; the forward states after each rotation are stored
(memory B * 4**n * 16 bytes per parameter).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import ROTATIONS, Circuit, apply_pauli, apply_rotation, bit_table


@dataclass(frozen=True)
class Noise:
    p1: float = 0.0      # depolarizing after each 1q gate
    p2: float = 0.0      # 2q depolarizing after each CZ
    g1: float = 0.0      # amplitude damping after each 1q gate
    g2: float = 0.0      # amplitude damping on both qubits after each CZ

    @property
    def is_noiseless(self):
        return not (self.p1 or self.p2 or self.g1 or self.g2)


def density_from_states(psi: np.ndarray) -> np.ndarray:
    psi = np.atleast_2d(psi)
    return (psi[:, :, None] * psi.conj()[:, None, :]).reshape(psi.shape[0], -1)


# --------------------------------------------------------------------------
# Single-qubit channels as linear maps on the 2x2 block of qubit q
# --------------------------------------------------------------------------
def _channel_1q(rho, n, q, kind, g, adjoint=False):
    B = rho.shape[0]
    a, c = 2 ** q, 2 ** (n - q - 1)
    v = rho.reshape(B, a, 2, c, a, 2, c)
    r00, r11 = v[:, :, 0, :, :, 0, :], v[:, :, 1, :, :, 1, :]
    out = np.empty_like(v)
    if kind == "depol":                      # self-adjoint (Pauli channel)
        out[:, :, 0, :, :, 0, :] = (1 - g / 2) * r00 + (g / 2) * r11
        out[:, :, 1, :, :, 1, :] = (g / 2) * r00 + (1 - g / 2) * r11
        off = 1 - g
    elif kind == "damp":
        if adjoint:
            out[:, :, 0, :, :, 0, :] = r00
            out[:, :, 1, :, :, 1, :] = (1 - g) * r11 + g * r00
        else:
            out[:, :, 0, :, :, 0, :] = r00 + g * r11
            out[:, :, 1, :, :, 1, :] = (1 - g) * r11
        off = np.sqrt(1 - g)
    else:  # pragma: no cover
        raise ValueError(kind)
    out[:, :, 0, :, :, 1, :] = off * v[:, :, 0, :, :, 1, :]
    out[:, :, 1, :, :, 0, :] = off * v[:, :, 1, :, :, 0, :]
    return out.reshape(rho.shape)


def _depol_2q(rho, n, a, b, lam):
    """(1 - lam) rho + lam * I/4 (x) Tr_ab rho = (1 - lam) rho + lam T_a T_b rho,
    with T_q the fully depolarizing channel on q. Self-adjoint."""
    full = _channel_1q(_channel_1q(rho, n, a, "depol", 1.0), n, b, "depol", 1.0)
    return (1 - lam) * rho + lam * full


class DensityCircuit:
    """A ``sim.Circuit`` executed on density matrices with a ``Noise`` model."""

    def __init__(self, circuit: Circuit, noise: Noise = Noise()):
        self.circ, self.noise = circuit, noise
        self.n, self.n_params = circuit.n, circuit.n_params
        bits = bit_table(self.n)
        self._cz = {}
        for kind, qubits, _ in circuit.ops:
            if kind == "cz" and qubits not in self._cz:
                i, j = qubits
                d = 1.0 - 2.0 * (bits[:, i] & bits[:, j])
                self._cz[qubits] = np.outer(d, d).ravel()
        self._diag = np.arange(2 ** self.n) * (2 ** self.n + 1)

    # -- one gate and the noise after it ----------------------------------
    def _gate(self, rho, op, theta, inverse=False):
        kind, q, idx = op
        if kind == "cz":
            return rho * self._cz[q]
        t = theta[..., idx]
        t = -t if inverse else t
        rho = apply_rotation(rho, 2 * self.n, kind, q, t)
        return apply_rotation(rho, 2 * self.n, kind, self.n + q, t if kind == "ry" else -t)

    def _noise(self, rho, op, adjoint=False):
        nz, n = self.noise, self.n
        kind, q, _ = op
        # forward: depolarizing, then amplitude damping; the adjoint map applies
        # the adjoint channels in reverse order
        if kind == "cz":
            a, b = q
            steps = [("depol2", nz.p2, None), ("damp", nz.g2, a), ("damp", nz.g2, b)]
        else:
            steps = [("depol", nz.p1, q), ("damp", nz.g1, q)]
        for k, g, qq in (steps[::-1] if adjoint else steps):
            if not g:
                continue
            if k == "depol2":
                rho = _depol_2q(rho, n, q[0], q[1], g)
            else:
                rho = _channel_1q(rho, n, qq, k, g, adjoint)
        return rho

    # -- public API --------------------------------------------------------
    def run(self, theta, rho0):
        theta = np.asarray(theta, float)
        rho = np.array(rho0, dtype=complex, copy=True)
        for op in self.circ.ops:
            rho = self._noise(self._gate(rho, op, theta), op)
        return rho

    def probabilities(self, theta, rho0):
        return np.real(self.run(theta, rho0)[:, self._diag])

    def expval_and_grad(self, theta, rho0, obs_diag):
        """``(f, g)``: f[b] = Tr[O rho_b], g[b, k] = d f[b] / d theta_k."""
        theta = np.asarray(theta, float)
        obs = np.asarray(obs_diag, float)
        rho = np.array(rho0, dtype=complex, copy=True)
        B = rho.shape[0]
        stored = []
        for op in self.circ.ops:
            rho = self._gate(rho, op, theta)
            if op[0] in ROTATIONS:
                stored.append(rho)
            rho = self._noise(rho, op)
        f = np.real(rho[:, self._diag]) @ obs
        lam = np.zeros_like(rho)
        lam[:, self._diag] = obs
        grad = np.zeros((B, self.n_params))
        for op in reversed(self.circ.ops):
            lam = self._noise(lam, op, adjoint=True)
            kind, q, idx = op
            if kind in ROTATIONS:
                p_rho = apply_pauli(stored.pop(), 2 * self.n, kind, q)
                grad[:, idx] += np.imag(np.sum(np.conj(lam) * p_rho, axis=1))
            lam = self._gate(lam, op, theta, inverse=True)
        return f, grad
