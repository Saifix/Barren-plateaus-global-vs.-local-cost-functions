"""The same circuits in Qiskit, for shot-based (Aer), noisy and hardware runs.

Qubit q here is Qiskit qubit q. Qiskit bitstrings are little-endian
(rightmost character = qubit 0); ``readout_from_counts`` accounts for that.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector


def hea_circuit(n: int, L: int, encode: bool = True, measure: bool = True):
    """Returns ``(circuit, x_params, theta_params)``; ``x_params`` is None if not encoding."""
    qc = QuantumCircuit(n)
    x = ParameterVector("x", n) if encode else None
    t = ParameterVector("t", 2 * n * L)
    if encode:
        for q in range(n):
            qc.ry(x[q], q)
    p = 0
    for _ in range(L):
        for q in range(n):
            qc.ry(t[p], q)
            qc.rz(t[p + 1], q)
            p += 2
        for q in range(0, n - 1, 2):
            qc.cz(q, q + 1)
        for q in range(1, n - 1, 2):
            qc.cz(q, q + 1)
    if measure:
        qc.measure_all()
    return qc, x, t


def readout_from_counts(counts: dict, n: int, readout: str, shots: int) -> float:
    if readout == "global":
        return counts.get("0" * n, 0) / shots
    zeros = np.zeros(n)
    for bitstring, c in counts.items():
        bits = bitstring.replace(" ", "")[::-1]  # bits[q] = qubit q
        zeros += c * np.array([b == "0" for b in bits], dtype=float)
    return float(zeros.mean() / shots)


def sampled_scores(sim, n, L, X, theta, readout, shots=2048, seed=0):
    """Class scores for every row of X, estimated from ``shots`` samples on ``sim``."""
    qc, xp, tp = hea_circuit(n, L)
    tqc = transpile(qc, sim, optimization_level=0)
    binds = {p: [float(theta[i])] * len(X) for i, p in enumerate(tp)}
    binds.update({p: [float(v) for v in X[:, i]] for i, p in enumerate(xp)})
    res = sim.run(tqc, shots=shots, parameter_binds=[binds], seed_simulator=seed).result()
    counts = [res.get_counts(i) for i in range(len(X))]
    return np.array([readout_from_counts(c, n, readout, shots) for c in counts])


def resources(n: int, L: int) -> dict:
    qc, _, _ = hea_circuit(n, L, measure=False)
    ops = qc.count_ops()
    return dict(qubits=n, layers=L, params=2 * n * L, cz=int(ops.get("cz", 0)),
                one_qubit_gates=int(ops.get("ry", 0) + ops.get("rz", 0)),
                depth=qc.depth(),
                circuits_per_gradient_ps=2 * (2 * n * L))  # parameter-shift, per sample
