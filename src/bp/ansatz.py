"""Hardware-efficient ansatz, cost observables and initializations.

Layer l (time order): R_Y then R_Z on every qubit, then CZ on (0,1),(2,3),...
followed by CZ on (1,2),(3,4),... (brickwork). Parameter index of
R_Y on qubit q in layer l is 2nl + 2q; R_Z is 2nl + 2q + 1.
"""
from __future__ import annotations

import math

import numpy as np

from .sim import Circuit, bit_table


def depth_log(n: int) -> int:
    """L = ceil(log2 n), at least 1."""
    return max(1, math.ceil(math.log2(n)))


def depth_for(n: int, label: str) -> int:
    return {"1": 1, "log": depth_log(n), "n": n}[label]


def hea_ops(n: int, L: int):
    ops, p = [], 0
    for _ in range(L):
        for q in range(n):
            ops.append(("ry", q, p))
            ops.append(("rz", q, p + 1))
            p += 2
        ops += [("cz", (q, q + 1), None) for q in range(0, n - 1, 2)]
        ops += [("cz", (q, q + 1), None) for q in range(1, n - 1, 2)]
    return ops, p


def hea(n: int, L: int) -> Circuit:
    return Circuit(n, *hea_ops(n, L))


def n_cz(n: int, L: int) -> int:
    return (n - 1) * L


def structurally_zero_params(n: int, L: int) -> np.ndarray:
    """R_Z angles of the last layer: only diagonal gates (CZ) follow them, so they
    commute with the Z-basis measurement and their gradient is identically 0
    for every Z-diagonal cost. They are excluded from 'average over k'."""
    return np.array([2 * n * (L - 1) + 2 * q + 1 for q in range(n)])


def product_rx(n: int) -> Circuit:
    """Analytic sanity-check ansatz: R_X(theta_j) on each qubit, no entanglers."""
    return Circuit(n, [("rx", q, q) for q in range(n)], n)


# --------------------------------------------------------------------------
# Observables.  Both costs are C = 1 - <O> with a Z-diagonal O, so both come
# from the same computational-basis measurement.
# --------------------------------------------------------------------------
def global_obs(n: int) -> np.ndarray:
    """O_G = |0^n><0^n|   ->   C_G = 1 - P(0^n)."""
    o = np.zeros(2 ** n)
    o[0] = 1.0
    return o


def local_obs(n: int) -> np.ndarray:
    """O_L = (1/n) sum_j |0><0|_j   ->   C_L = 1 - (1/n) sum_j P(q_j = 0)."""
    return (1 - bit_table(n)).sum(axis=1) / n


def observable(n: int, cost: str) -> np.ndarray:
    return {"global": global_obs, "local": local_obs}[cost](n)


# --------------------------------------------------------------------------
# Initializations
# --------------------------------------------------------------------------
def init_uniform(rng, size):
    return rng.uniform(0.0, 2 * np.pi, size)


def init_reduced(rng, size, L: int, scale: float = np.pi):
    """theta ~ U[-a, a] with a = scale / sqrt(L)  (Wang et al. 2024)."""
    a = scale / math.sqrt(L)
    return rng.uniform(-a, a, size)


def init_params(kind: str, rng, size, L: int, scale: float = np.pi):
    """Dispatch for the 'uniform' and 'reduced' schemes. Identity blocks need
    the circuit layout, see ``init_identity_blocks``."""
    if kind == "uniform":
        return init_uniform(rng, size)
    if kind == "reduced":
        return init_reduced(rng, size, L, scale)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Identity blocks (Grant et al. 2019): the circuit is a block X(alpha) followed
# by a sub-circuit shaped like X^dagger with its own angles beta; beta = -alpha
# (in mirrored order) makes the block exactly the identity.
#
# X must start and end with a rotation layer. If it ended with a CZ layer (as
# an HEA layer does), the first gate of the X^dagger half would be the same CZ
# layer, and the two would cancel for *every* parameter value. With one-layer
# halves the whole circuit would then be a product circuit. So
#   X(alpha)       = R_1, CZ, R_2, CZ, ..., CZ, R_D        (D rotation layers)
#   X^dag-shaped   = R'_D, CZ, ..., CZ, R'_1, with R' = R_Z then R_Y
# where R_i = R_Y then R_Z on every qubit and CZ is the brickwork of the HEA.
# X is the D-layer HEA without its final CZ layer, which is diagonal and so
# invisible to every Z-basis cost. One block has 2D rotation layers, 2(D-1) CZ
# layers and 4nD parameters. Parameter layout: rotation layer i (0-based,
# time order) owns indices 2n*i ... 2n*i + 2n - 1; within a layer, R_Y on qubit
# q is 2q and R_Z is 2q + 1, in both halves.
# All three initializations in the H3 experiments use this circuit, so they
# differ only in the starting angles.
# --------------------------------------------------------------------------
def _brickwork(n):
    return ([("cz", (q, q + 1), None) for q in range(0, n - 1, 2)]
            + [("cz", (q, q + 1), None) for q in range(1, n - 1, 2)])


def ib_hea_ops(n: int, D: int, blocks: int = 1):
    ops, layer = [], 0
    for _ in range(blocks):
        for i in range(D):                       # X(alpha)
            if i:
                ops += _brickwork(n)
            p = 2 * n * layer
            for q in range(n):
                ops += [("ry", q, p + 2 * q), ("rz", q, p + 2 * q + 1)]
            layer += 1
        for i in range(D):                       # X(beta)^dagger shape
            if i:
                ops += _brickwork(n)[::-1]
            p = 2 * n * layer
            for q in range(n):
                ops += [("rz", q, p + 2 * q + 1), ("ry", q, p + 2 * q)]
            layer += 1
    return ops, 2 * n * layer


def ib_hea(n: int, D: int, blocks: int = 1) -> Circuit:
    return Circuit(n, *ib_hea_ops(n, D, blocks))


def ib_cz_layers(D: int, blocks: int = 1) -> int:
    return 2 * (D - 1) * blocks


def init_identity_blocks(rng, size, n: int, D: int, blocks: int = 1):
    """Identity-block initialization for ``ib_hea(n, D, blocks)``.

    The X half of every block gets uniform angles in [0, 2pi); rotation layer
    D-1-i of the X^dagger half gets minus the angles of layer i of the X half,
    so every block, and the whole circuit, is exactly the identity.
    ``size`` is ``P`` or ``(B, P)`` with ``P = 4nD * blocks``.
    """
    theta = np.zeros(size)
    w = 2 * n
    for b in range(blocks):
        base = 2 * D * b
        for i in range(D):
            a = rng.uniform(0.0, 2 * np.pi, theta[..., :w].shape)
            theta[..., (base + i) * w:(base + i + 1) * w] = a
            j = base + 2 * D - 1 - i
            theta[..., j * w:(j + 1) * w] = -a
    return theta


def make_init(kind: str, rng, size, n: int, D: int, blocks: int = 1, scale: float = np.pi):
    """Initial angles for ``ib_hea(n, D, blocks)``. The reduced domain uses the
    number of rotation layers, 2*D*blocks, as its depth L."""
    if kind == "identity":
        return init_identity_blocks(rng, size, n, D, blocks)
    return init_params(kind, rng, size, 2 * D * blocks, scale)


def trivial_params(circ: Circuit) -> np.ndarray:
    """Indices of R_Z angles followed on their qubit only by diagonal gates.

    Such a gate commutes with everything after it and with a Z-basis
    measurement, so its gradient is identically zero for every cost here
    (generalizes ``structurally_zero_params`` to any gate order)."""
    out = []
    for i, (kind, q, idx) in enumerate(circ.ops):
        if kind != "rz":
            continue
        later = circ.ops[i + 1:]
        if all(k in ("rz", "cz") for k, qs, _ in later
               if (qs == q if k != "cz" else q in qs)):
            out.append(idx)
    return np.array(sorted(out), dtype=int)


def layer_of_param(n: int, P: int) -> np.ndarray:
    """Rotation-layer index (0-based) of each parameter (HEA and IB-HEA)."""
    return np.arange(P) // (2 * n)
