"""Noise: gradient variance under unital (depolarizing) and non-unital
(amplitude-damping) gate noise, both costs, exact density-matrix gradients.

Standard HEA, |0...0> input, uniform init, NOISE_DRAWS draws per cell; within
an (n, L) cell the same parameter draws are used for every noise model and
both costs (paired). Noise after every gate (bp.density): strength P2 on CZ
(two-qubit depolarizing, or amplitude damping on both qubits), P2/10 after
each one-qubit gate. Three studies:
  n      n in NOISE_NS, L = ceil(log2 n), P2 = NOISE_P2
  depth  n = 6, L in NOISE_DEPTHS, P2 = NOISE_P2   (noise-induced decay with L)
  p      n = 6, L = 6, P2 in NOISE_STRENGTHS
The noiseless reference uses the state-vector simulator (identical result,
see tests/test_density.py).

results/noise_variance.csv, results/noise_variance_layers.csv
"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bp import SEED
from bp.ansatz import depth_log, hea, init_uniform, layer_of_param, observable, structurally_zero_params
from bp.config import (N_BOOT, NOISE_DEPTH_N, NOISE_DEPTHS, NOISE_DRAWS, NOISE_NS, NOISE_P2,
                       NOISE_STRENGTHS, VAR_K_FIXED)
from bp.data import ROOT
from bp.density import DensityCircuit, Noise, density_from_states
from bp.sim import zero_state
from bp.variance import bootstrap, variance_stats

MAX_WORKERS = 14
MEM_BUDGET = 3e8          # bytes of stored density matrices per batch


def noise_for(model, p2):
    if model == "depolarizing":
        return Noise(p1=p2 / 10, p2=p2)
    if model == "amplitude_damping":
        return Noise(g1=p2 / 10, g2=p2)
    return Noise()


def job(args):
    study, n, L, model, p2 = args
    circ = hea(n, L)
    P = circ.n_params
    theta = init_uniform(np.random.default_rng([SEED, n, L, 11]), (NOISE_DRAWS, P))
    grads = {c: np.empty((NOISE_DRAWS, P)) for c in ("global", "local")}
    costs = {c: np.empty(NOISE_DRAWS) for c in grads}
    t0 = time.time()
    if model == "noiseless":
        for s in range(0, NOISE_DRAWS, 50):
            th = theta[s:s + 50]
            for c in grads:
                f, g = circ.expval_and_grad(th, zero_state(n, len(th)), observable(n, c))
                grads[c][s:s + 50], costs[c][s:s + 50] = -g, 1 - f
    else:
        dc = DensityCircuit(circ, noise_for(model, p2))
        batch = max(1, int(MEM_BUDGET / (P * 16 * 4 ** n)))
        for s in range(0, NOISE_DRAWS, batch):
            th = theta[s:s + batch]
            rho0 = density_from_states(zero_state(n, len(th)))
            for c in grads:
                f, g = dc.expval_and_grad(th, rho0, observable(n, c))
                grads[c][s:s + batch], costs[c][s:s + batch] = -g, 1 - f
    keep = np.setdiff1d(np.arange(P), structurally_zero_params(n, L))
    layers = layer_of_param(n, P)
    rows, lay = [], []
    for c, g in grads.items():
        v_k, v_mean = variance_stats(g, n, L, VAR_K_FIXED, keep)
        bs = bootstrap(g, n, L, np.random.default_rng([SEED, n, L, 7]), N_BOOT, VAR_K_FIXED, keep)
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        rows.append(dict(study=study, n=n, L=L, model=model, p2=p2, cost=c, draws=NOISE_DRAWS,
                         var_k=v_k, var_k_lo=lo[0], var_k_hi=hi[0],
                         var_mean=v_mean, var_mean_lo=lo[1], var_mean_hi=hi[1],
                         mean_cost=float(costs[c].mean()), std_cost=float(costs[c].std(ddof=1))))
        v = g.var(axis=0, ddof=1)
        for l in range(L):
            idx = np.intersect1d(np.where(layers == l)[0], keep)
            lay.append(dict(study=study, n=n, L=L, model=model, p2=p2, cost=c, layer=l,
                            var_mean=float(v[idx].mean())))
    print(f"{study:>5} n={n} L={L:>2} {model:>17} p2={p2:<6} Var_G={rows[0]['var_k']:.2e} "
          f"Var_L={rows[1]['var_k']:.2e} ({time.time() - t0:.0f}s)", flush=True)
    return rows, lay


def main():
    models = ("noiseless", "depolarizing", "amplitude_damping")
    jobs = [("n", n, depth_log(n), m, NOISE_P2) for n in NOISE_NS for m in models]
    jobs += [("depth", NOISE_DEPTH_N, L, m, NOISE_P2) for L in NOISE_DEPTHS for m in models]
    jobs += [("p", NOISE_DEPTH_N, 6, m, p) for p in NOISE_STRENGTHS for m in models[1:]]
    jobs += [("p", NOISE_DEPTH_N, 6, "noiseless", 0.0)]
    jobs.sort(key=lambda j: -(4 ** j[1]) * j[2])
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        res = list(ex.map(job, jobs))
    out = ROOT / "results"
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    df.to_csv(out / "noise_variance.csv", index=False)
    pd.DataFrame([r for _, lay in res for r in lay]).to_csv(out / "noise_variance_layers.csv", index=False)
    print(df[df.study == "n"].pivot_table(index=["n"], columns=["cost", "model"], values="var_k")
          .to_string(float_format="%.2e"))


if __name__ == "__main__":
    main()
