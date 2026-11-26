"""Analytical gradient variance of a random-angle circuit, via Pauli paths.

For a circuit of Pauli rotations R_P(t) = exp(-i t P / 2) with independent,
uniformly distributed angles, Clifford gates (CZ) and Pauli-diagonal noise,
the variance over angles of d<O>/dt_k has an exact closed form (cf. Letcher
et al., Quantum 8, 1484 (2024)):

  * Expand the observable in Pauli strings and move it backwards through the
    circuit (Heisenberg picture). A rotation leaves a string that commutes with
    its generator unchanged and sends one that anticommutes to
    cos(t) Q + sin(t) Q', i.e. it splits a path in two.
  * Averaged over a uniform angle, cos^2 = sin^2 = 1/2 and every cross term
    between two different paths vanishes, so E[<O>^2] (and E[(d<O>/dt_k)^2])
    is a sum of SQUARED path weights. The squared weights therefore evolve by a
    linear, stochastic-like map on the 4^n Pauli strings: this module
    propagates that vector exactly.
  * The derivative at rotation k keeps only the strings that anticommute with
    its generator (the commuting ones do not depend on t_k); for those the
    factor is again 1/2 per branch. E[d<O>/dt_k] = 0, so this is the variance.
  * The initial state |0...0> gives <0|P|0> = 1 for P in {I, Z}^n, else 0.
  * Depolarizing noise (Qiskit convention, rho -> (1 - lam) rho + lam I/d (x)
    Tr rho) multiplies every Pauli string that is non-identity on its qubits by
    (1 - lam), i.e. a squared weight by (1 - lam)^2. This is exact.
  * Amplitude damping is non-unital: besides damping X, Y by sqrt(1 - g) and Z
    by (1 - g), it moves weight from Z to the identity, and that part creates
    cross terms. ``damping="unital_part"`` keeps only the diagonal damping and
    drops the Z -> I transfer, so the model predicts what damping would do if it
    were unital; the difference to the simulation isolates the non-unital part.

The circuit and noise placement are those of ``bp.ansatz.hea`` and
``bp.density.DensityCircuit``: noise after every rotation on its qubit and after
every CZ on both qubits. Cost is O(#ops * 4^n); n <= 12 is practical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ansatz import hea_ops, structurally_zero_params

# Pauli index on each axis: 0 = I, 1 = X, 2 = Y, 3 = Z.  (x, z) bits per index:
_XZ = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
_IDX = {v: k for k, v in _XZ.items()}

# Squared-weight transfer of one uniformly random rotation, as M[out, in].
_ROT = {
    "rx": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, .5, .5], [0, 0, .5, .5]]),
    "ry": np.array([[1, 0, 0, 0], [0, .5, 0, .5], [0, 0, 1, 0], [0, .5, 0, .5]]),
    "rz": np.array([[1, 0, 0, 0], [0, .5, .5, 0], [0, .5, .5, 0], [0, 0, 0, 1]]),
}
# Strings on the rotated qubit that anticommute with its generator.
_ANTI = {"rx": [2, 3], "ry": [1, 3], "rz": [1, 2]}
_DER = {k: m * np.isin(np.arange(4), _ANTI[k])[None, :] for k, m in _ROT.items()}


def _cz_perm() -> np.ndarray:
    """CZ conjugation of a two-qubit Pauli string (a, b) -> flat index 4a+b."""
    perm = np.empty(16, dtype=int)
    for a in range(4):
        for b in range(4):
            xa, za = _XZ[a]
            xb, zb = _XZ[b]
            perm[4 * a + b] = 4 * _IDX[(xa, za ^ xb)] + _IDX[(xb, zb ^ xa)]
    return perm


_CZ = _cz_perm()


@dataclass(frozen=True)
class PathNoise:
    """Same parameters as ``bp.density.Noise``."""
    p1: float = 0.0
    p2: float = 0.0
    g1: float = 0.0
    g2: float = 0.0


def _on_axis(w, M, q):
    return np.moveaxis(np.tensordot(M, w, axes=([1], [q])), 0, q)


def _cz(w, a, b):
    n = w.ndim
    v = np.moveaxis(w, (a, b), (0, 1)).reshape(16, -1)
    v = v[_CZ]                                     # CZ is an involution: perm^-1 = perm
    return np.moveaxis(v.reshape((4, 4) + (4,) * (n - 2)), (0, 1), (a, b))


def _depol1(w, q, lam):
    d = np.array([1.0] + [(1 - lam) ** 2] * 3)
    return _on_axis(w, np.diag(d), q)


def _depol2(w, a, b, lam):
    n = w.ndim
    f = np.full(16, (1 - lam) ** 2)
    f[0] = 1.0
    v = np.moveaxis(w, (a, b), (0, 1)).reshape(16, -1) * f[:, None]
    return np.moveaxis(v.reshape((4, 4) + (4,) * (n - 2)), (0, 1), (a, b))


def _damp(w, q, g):
    return _on_axis(w, np.diag([1.0, 1 - g, 1 - g, (1 - g) ** 2]), q)


def _noise(w, op, nz: PathNoise):
    kind, q, _ = op
    if kind == "cz":
        a, b = q
        if nz.p2:
            w = _depol2(w, a, b, nz.p2)
        if nz.g2:
            w = _damp(_damp(w, a, nz.g2), b, nz.g2)
    else:
        if nz.p1:
            w = _depol1(w, q, nz.p1)
        if nz.g1:
            w = _damp(w, q, nz.g1)
    return w


def _gate(w, op):
    kind, q, _ = op
    return _cz(w, *q) if kind == "cz" else _on_axis(w, _ROT[kind], q)


def observable_weights(n: int, cost: str) -> np.ndarray:
    """Squared Pauli coefficients of the cost's observable (sign and constant
    dropped: they do not change the gradient variance)."""
    w = np.zeros((4,) * n)
    if cost == "global":                     # |0><0|^n = 2^-n sum_S Z_S
        idx = np.ix_(*[[0, 3]] * n)
        w[idx] = 4.0 ** (-n)
    elif cost == "local":                    # (1/n) sum_j (I + Z_j) / 2
        for j in range(n):
            ix = [0] * n
            ix[j] = 3
            w[tuple(ix)] = 1.0 / (4 * n * n)
    else:
        raise ValueError(cost)
    return w


def initial_weights(n: int) -> np.ndarray:
    """<0|P|0>^2: 1 on {I, Z}^n."""
    w = np.zeros((4,) * n)
    w[np.ix_(*[[0, 3]] * n)] = 1.0
    return w


def gradient_variances(n: int, L: int, cost: str, noise: PathNoise = PathNoise(),
                       params=None, ops=None, n_params=None) -> dict:
    """Var over uniform angles of dC/dt_k for the HEA(n, L) (or the given ops).

    params: parameter indices to evaluate (default: all). Returns {k: Var}.
    Memory: one 4^n vector per requested parameter (the forward pass is stored).
    """
    if ops is None:
        ops, n_params = hea_ops(n, L)
    params = range(n_params) if params is None else list(params)
    want = set(params)
    # forward pass: f just before every requested rotation
    f, before = initial_weights(n), {}
    for op in ops:
        if op[0] != "cz" and op[2] in want:
            before[op[2]] = f
        f = _noise(_gate(f, op), op, noise)
    # backward pass: b just after every gate's noise, contracted on the fly
    b, out = observable_weights(n, cost), {}
    for op in reversed(ops):
        b = _noise(b, op, noise)
        if op[0] != "cz" and op[2] in want:
            out[op[2]] = float(np.sum(before.pop(op[2]) * _on_axis(b, _DER[op[0]], op[1])))
        b = _gate(b, op)
    return out


def variance_summary(n: int, L: int, cost: str, noise: PathNoise = PathNoise(), k_fixed: int = 0,
                     all_params: bool = True) -> dict:
    """var_k (parameter k_fixed) and var_mean (average over the parameters that
    are not structurally zero), matching ``bp.variance.variance_stats``."""
    _, P = hea_ops(n, L)
    keep = np.setdiff1d(np.arange(P), structurally_zero_params(n, L))
    params = keep if all_params else [k_fixed]
    v = gradient_variances(n, L, cost, noise, params=sorted(set(params) | {k_fixed}))
    out = dict(var_k=v[k_fixed])
    if all_params:
        out["var_mean"] = float(np.mean([v[k] for k in keep]))
    return out


def mean_pauli_weight(n: int, L: int, cost: str, noise: PathNoise = PathNoise()) -> list:
    """Average Pauli weight (number of non-identity qubits) of the backward-
    propagated observable after each layer, last layer first. A local cost's
    weight is capped by its light cone; a global cost starts at about n/2."""
    ops, _ = hea_ops(n, L)
    weight = np.zeros((4,) * n)
    for q in range(n):
        weight = weight + (np.arange(4) > 0).reshape([-1 if i == q else 1 for i in range(n)])
    b, per_layer, done = observable_weights(n, cost), [], 0
    per_op = (2 * n) + (n - 1)                 # rotations + CZs of one layer
    for i, op in enumerate(reversed(ops)):
        b = _gate(_noise(b, op, noise), op)
        if (i + 1) % per_op == 0:
            nz = b.copy()
            nz.flat[0] = 0.0                         # the identity carries no gradient
            per_layer.append(float((nz * weight).sum() / nz.sum()))
            done += 1
    return per_layer
