"""Noise model of an IBM device: gradient variance on the FakeFez noise model.

Qiskit Aer, density-matrix method, with the noise model of FakeFez (IBM
Heron r2, native CZ): per-gate depolarizing and thermal-relaxation errors from
the device calibration, plus readout assignment errors. Exact output
probabilities (no shot sampling), so the measured variance is a property of
the noisy landscape; the shot-noise floor for HW_SHOTS shots is computed
separately from the same probabilities.

Circuit: standard HEA, |0...0> input, L = ceil(log2 n), n in FAKE_NS, placed on
the best contiguous segment of the linear chain of physical qubits 0..15 (all
brickwork CZs are native couplings, so no SWAPs). Parameter-shift gradients
for two angles: theta_1 (R_Y, qubit 0, first layer; as in every other
experiment) and the R_Y on qubit 0 in the last layer. The noiseless reference
is computed at the same draws.

results/fake_backend.csv, results/fake_backend_layout.json
"""
import json
import time

import numpy as np
import pandas as pd
from qiskit import transpile

from bp import SEED
from bp.ansatz import depth_log, hea, init_uniform, observable
from bp.config import FAKE_BACKEND, FAKE_DRAWS, FAKE_NS, HW_SHOTS, N_BOOT
from bp.data import results_dir
from bp.qiskit_backend import hea_circuit
from bp.sim import zero_state


def pick_layout(backend, n, chain=range(16)):
    """Contiguous window of the chain with the lowest summed CZ + readout error."""
    chain = list(chain)
    t = backend.target
    best = None
    for s in range(len(chain) - n + 1):
        qs = chain[s:s + n]
        cz = sum(t["cz"][(a, b)].error if (a, b) in t["cz"] else t["cz"][(b, a)].error
                 for a, b in zip(qs, qs[1:]))
        ro = sum(t["measure"][(q,)].error for q in qs)
        if best is None or cz + ro < best[0]:
            best = (cz + ro, qs, cz, ro)
    return best[1], dict(physical_qubits=best[1], sum_cz_error=best[2], sum_readout_error=best[3])


def readout_matrices(backend, qubits):
    props = backend.properties()
    mats = []
    for q in qubits:
        p10 = props.qubit_property(q, "prob_meas1_prep0")[0]
        p01 = props.qubit_property(q, "prob_meas0_prep1")[0]
        mats.append(np.array([[1 - p10, p01], [p10, 1 - p01]]))   # [measured, prepared]
    return mats


def apply_readout(probs, mats):
    """probs: (B, 2**n) big-endian (qubit 0 = most significant)."""
    B, n = probs.shape[0], len(mats)
    t = probs.reshape((B,) + (2,) * n)
    for q, A in enumerate(mats):
        t = np.moveaxis(np.tensordot(A, t, axes=([1], [q + 1])), 0, q + 1)
    return t.reshape(B, -1)


def boot_ci(x, rng):
    m = len(x)
    bs = np.array([x[rng.integers(0, m, m)].var(ddof=1) for _ in range(N_BOOT)])
    return np.percentile(bs, [2.5, 97.5])


def main():
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    import qiskit_ibm_runtime.fake_provider as fp
    backend = getattr(fp, FAKE_BACKEND)()
    nm = NoiseModel.from_backend(backend)
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    rows, layouts = [], {}
    for n in FAKE_NS:
        t0 = time.time()
        L = depth_log(n)
        circ = hea(n, L)
        P = circ.n_params
        ks = {"first": 0, "last": 2 * n * (L - 1)}
        theta = init_uniform(np.random.default_rng([SEED, n, L, 13]), (FAKE_DRAWS, P))
        layout, info = pick_layout(backend, n)
        layouts[n] = info
        qc, _, tp = hea_circuit(n, L, encode=False, measure=False)
        tqc = transpile(qc, backend, initial_layout=layout, optimization_level=0,
                        seed_transpiler=SEED)
        phys = [tqc.layout.initial_layout[qc.qubits[q]] for q in range(n)]
        tqc.save_probabilities(qubits=phys[::-1])    # Aer: first listed qubit = least significant
        # parameter sets: for each draw and each k, theta +- pi/2 e_k
        sets = []
        for d in range(FAKE_DRAWS):
            for k in ks.values():
                for sgn in (1, -1):
                    th = theta[d].copy()
                    th[k] += sgn * np.pi / 2
                    sets.append(th)
        sets = np.array(sets)
        binds = {p: sets[:, i].tolist() for i, p in enumerate(tp)}
        res = sim.run(tqc, parameter_binds=[binds], seed_simulator=SEED).result()
        probs = np.array([res.data(i)["probabilities"] for i in range(len(sets))])
        probs_ro = apply_readout(probs, readout_matrices(backend, phys))
        exact = circ.probabilities(sets, zero_state(n, len(sets)))
        for c in ("global", "local"):
            o = observable(n, c)
            for model, pr in (("noiseless", exact), ("fake_no_readout", probs), ("fake", probs_ro)):
                f = (pr @ o).reshape(FAKE_DRAWS, len(ks), 2)
                g = -(f[..., 0] - f[..., 1]) / 2                     # dC/dtheta_k
                s2 = (pr @ o ** 2 - (pr @ o) ** 2).reshape(FAKE_DRAWS, len(ks), 2)
                floor = (s2[..., 0] + s2[..., 1]).mean(axis=0) / (4 * HW_SHOTS)
                for j, (kname, k) in enumerate(ks.items()):
                    lo, hi = boot_ci(g[:, j], np.random.default_rng([SEED, n, j, 7]))
                    rows.append(dict(n=n, L=L, k=kname, k_index=k, cost=c, model=model,
                                     draws=FAKE_DRAWS, var=g[:, j].var(ddof=1), var_lo=lo, var_hi=hi,
                                     shot_floor=floor[j], mean_cost=1 - f[:, j].mean()))
        print(f"n={n} L={L} layout={phys} ({time.time() - t0:.0f}s)", flush=True)
    out = results_dir(2)
    df = pd.DataFrame(rows)
    df.to_csv(out / "fake_backend.csv", index=False)
    (out / "fake_backend_layout.json").write_text(json.dumps(
        dict(backend=FAKE_BACKEND, layouts={str(k): v for k, v in layouts.items()}), indent=1))
    print(df.pivot_table(index=["n", "k"], columns=["cost", "model"], values="var")
          .to_string(float_format="%.2e"))


if __name__ == "__main__":
    main()
