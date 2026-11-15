"""Batched state-vector simulator with exact adjoint gradients.

Why a small custom simulator instead of a framework:
  * every quantity we need is the expectation of a *diagonal* observable
    (both costs and both classifier readouts are functions of Z-basis
    probabilities), so the adjoint method gives all 2nL partial derivatives
    for the price of roughly two forward passes;
  * the batch axis can carry either many data points with shared parameters
    (classifier training) or many independent parameter draws (gradient-
    variance sweeps), with the same code path;
  * it is short enough to verify line by line. ``tests/`` checks it against
    Qiskit's Statevector, finite differences, the parameter-shift rule and the
    closed-form product-ansatz variances.

Conventions
-----------
* A state batch has shape ``(B, 2**n)``. Qubit 0 is the *most significant* bit
  of the basis-state index (big-endian). Qiskit uses the opposite convention;
  ``qiskit_backend`` handles the conversion.
* An op is a tuple ``(kind, qubits, param_index)`` with ``kind`` in
  ``{"rx", "ry", "rz", "cz"}``. Rotations are ``R_P(t) = exp(-i t P / 2)``.
* ``theta`` is either shape ``(P,)`` (shared by the whole batch) or
  ``(B, P)`` (one parameter vector per batch element).
"""
from __future__ import annotations

import numpy as np

ROTATIONS = ("rx", "ry", "rz")


def bit_table(n: int) -> np.ndarray:
    """``(2**n, n)`` array; column q is the value of qubit q in each basis state."""
    idx = np.arange(2 ** n)
    return (idx[:, None] >> (n - 1 - np.arange(n))) & 1


def zero_state(n: int, batch: int = 1) -> np.ndarray:
    psi = np.zeros((batch, 2 ** n), dtype=complex)
    psi[:, 0] = 1.0
    return psi


def product_state(amp0: np.ndarray, amp1: np.ndarray) -> np.ndarray:
    """Batch of product states sum_b amp0/amp1 per qubit. Inputs ``(B, n)``."""
    amp0 = np.asarray(amp0, dtype=complex)
    amp1 = np.asarray(amp1, dtype=complex)
    batch, n = amp0.shape
    state = np.ones((batch, 1), dtype=complex)
    for q in range(n):
        qubit = np.stack([amp0[:, q], amp1[:, q]], axis=1)  # (B, 2)
        state = (state[:, :, None] * qubit[:, None, :]).reshape(batch, -1)
    return state


def angle_encode(x: np.ndarray) -> np.ndarray:
    """``R_Y(x_j)`` on every qubit of |0...0>, one data point per row of ``x``."""
    x = np.atleast_2d(x)
    return product_state(np.cos(x / 2), np.sin(x / 2))


# --------------------------------------------------------------------------
# Gate kernels
# --------------------------------------------------------------------------
def _halves(state: np.ndarray, n: int, q: int):
    v = state.reshape(state.shape[0], 2 ** q, 2, 2 ** (n - q - 1))
    return v, v[:, :, 0, :], v[:, :, 1, :]


def _col(angle, batch: int) -> np.ndarray:
    """Broadcastable angle: scalar -> (1,1,1); (B,) -> (B,1,1)."""
    a = np.asarray(angle, dtype=float)
    return a.reshape(-1, 1, 1) if a.ndim else a.reshape(1, 1, 1)


def apply_rotation(state, n, kind, q, angle):
    v, a, b = _halves(state, n, q)
    t = _col(angle, state.shape[0]) / 2
    c, s = np.cos(t), np.sin(t)
    out = np.empty_like(v)
    if kind == "ry":
        out[:, :, 0, :] = c * a - s * b
        out[:, :, 1, :] = s * a + c * b
    elif kind == "rx":
        out[:, :, 0, :] = c * a - 1j * s * b
        out[:, :, 1, :] = -1j * s * a + c * b
    elif kind == "rz":
        ph = np.exp(-1j * t)
        out[:, :, 0, :] = ph * a
        out[:, :, 1, :] = np.conj(ph) * b
    else:  # pragma: no cover
        raise ValueError(kind)
    return out.reshape(state.shape)


def apply_pauli(state, n, kind, q):
    """Apply the generator P of ``R_P`` (X for rx, Y for ry, Z for rz)."""
    v, a, b = _halves(state, n, q)
    out = np.empty_like(v)
    if kind == "rx":
        out[:, :, 0, :], out[:, :, 1, :] = b, a
    elif kind == "ry":
        out[:, :, 0, :], out[:, :, 1, :] = -1j * b, 1j * a
    elif kind == "rz":
        out[:, :, 0, :], out[:, :, 1, :] = a, -b
    else:  # pragma: no cover
        raise ValueError(kind)
    return out.reshape(state.shape)


class Circuit:
    """A fixed gate sequence with ``n_params`` trainable rotation angles."""

    def __init__(self, n: int, ops: list, n_params: int):
        self.n, self.ops, self.n_params = n, list(ops), n_params
        bits = bit_table(n)
        self._cz = {}
        for kind, qubits, _ in self.ops:
            if kind == "cz" and qubits not in self._cz:
                i, j = qubits
                self._cz[qubits] = 1.0 - 2.0 * (bits[:, i] & bits[:, j])

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _angle(theta, idx):
        return theta[..., idx]

    def _apply(self, state, op, theta, inverse=False):
        kind, qubits, idx = op
        if kind == "cz":
            return state * self._cz[qubits]
        angle = self._angle(theta, idx)
        return apply_rotation(state, self.n, kind, qubits, -angle if inverse else angle)

    # -- public API ------------------------------------------------------
    def run(self, theta, psi0) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        state = np.array(psi0, dtype=complex, copy=True)
        for op in self.ops:
            state = self._apply(state, op, theta)
        return state

    def probabilities(self, theta, psi0) -> np.ndarray:
        return np.abs(self.run(theta, psi0)) ** 2

    def expval_and_grad(self, theta, psi0, obs_diag, weights=None):
        """Expectation of a diagonal observable and its gradient (adjoint method).

        Returns ``(f, g)`` with ``f[b] = <psi_b|O|psi_b>`` and
        ``g[b, k] = w_b * d f[b] / d theta_k``. Sum ``g`` over the batch axis
        for a shared-parameter loss ``sum_b w_b f_b``. ``weights`` may be an
        array or a callable ``f -> w`` (chain rule for a loss of f, evaluated
        after the forward pass, so one forward pass serves both).
        """
        theta = np.asarray(theta, dtype=float)
        obs = np.asarray(obs_diag, dtype=float)
        phi = self.run(theta, psi0)
        f = (np.abs(phi) ** 2) @ obs
        if callable(weights):
            weights = weights(f)
        w = np.ones(phi.shape[0]) if weights is None else np.asarray(weights, float)
        lam = (w[:, None] * obs[None, :]) * phi
        grad = np.zeros((phi.shape[0], self.n_params))
        for op in reversed(self.ops):
            kind, qubits, idx = op
            if kind in ROTATIONS:
                p_phi = apply_pauli(phi, self.n, kind, qubits)
                grad[:, idx] += np.imag(np.sum(np.conj(lam) * p_phi, axis=1))
            phi = self._apply(phi, op, theta, inverse=True)
            lam = self._apply(lam, op, theta, inverse=True)
        return f, grad
