"""Noise-free gradient-variance sweep, uniform initialization (H1, H2).

Writes results/variance_sweep.csv (one row per n, L, cost) and
results/variance_decay_fits.csv (fitted decay rate b with bootstrap CI).
Exact state-vector gradients on purpose: they isolate the property of the
landscape from the error of estimating it. With N shots, a parameter-shift
estimate adds variance (p+(1-p+) + p-(1-p-)) / (4N); for the global cost
p ~ 2^-n, so this term shrinks with n too and is comparable to the signal at
the largest n. Finite-shot effects are studied separately (Aer, hardware).
"""
import time

import numpy as np
import pandas as pd

from bp import SEED
from bp.ansatz import depth_for
from bp.config import N_BOOT, VAR_DEPTHS, VAR_K_FIXED, VAR_NS, var_draws
from bp.data import results_dir
from bp.variance import bootstrap, fit_decay, sample_gradients, variance_stats

OUT = results_dir(1)
OUT.mkdir(exist_ok=True)

rows, boots = [], {}
for depth in VAR_DEPTHS:
    for n in VAR_NS:
        L = depth_for(n, depth)
        rng = np.random.default_rng([SEED, n, L])
        t0 = time.time()
        draws = var_draws(n)
        grads = sample_gradients(n, L, draws, rng)
        for cost, g in grads.items():
            v_k, v_mean = variance_stats(g, n, L, VAR_K_FIXED)
            bs = bootstrap(g, n, L, np.random.default_rng([SEED, n, L, 7]), N_BOOT, VAR_K_FIXED)
            boots[(depth, cost, n)] = bs
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            rows.append(dict(depth=depth, n=n, L=L, cost=cost, draws=draws,
                             var_k=v_k, var_k_lo=lo[0], var_k_hi=hi[0],
                             var_mean=v_mean, var_mean_lo=lo[1], var_mean_hi=hi[1]))
        print(f"depth={depth:>3} n={n:>2} L={L:>2}  "
              f"Var_G={rows[-2]['var_k']:.2e}  Var_L={rows[-1]['var_k']:.2e}  "
              f"({time.time() - t0:.1f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "variance_sweep.csv", index=False)

# Decay-rate fits  log Var = a - b n,  bootstrap CI by refitting on resampled variances
fits = []
for depth in VAR_DEPTHS:
    for cost in ("global", "local"):
        sub = df[(df.depth == depth) & (df.cost == cost)].sort_values("n")
        ns = sub.n.to_numpy()
        for j, stat in enumerate(("var_k", "var_mean")):
            _, b = fit_decay(ns, sub[stat])
            bb = np.array([fit_decay(ns, [boots[(depth, cost, n)][i, j] for n in ns])[1]
                           for i in range(N_BOOT)])
            fits.append(dict(depth=depth, cost=cost, stat=stat, b=b,
                             b_lo=np.percentile(bb, 2.5), b_hi=np.percentile(bb, 97.5)))
pd.DataFrame(fits).to_csv(OUT / "variance_decay_fits.csv", index=False)
print(pd.DataFrame(fits).round(3).to_string(index=False))
