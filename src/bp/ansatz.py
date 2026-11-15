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
