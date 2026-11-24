"""H3, part 1: initial gradient variance under three initializations.

Circuit: the identity-block HEA ib_hea(n, D), D = ceil(log2 n) (one block of
2D rotation layers, 2(D-1) CZ layers; see bp.ansatz). Target state:
|psi0> = V(theta*)^dagger |0...0> with a fresh teacher theta* ~ U[0, 2pi) per
draw, i.e. the training task of 08 averaged over targets. For every draw the
gradients of C_G and C_L are taken at the same (target, theta) pair.

Initializations: uniform U[0, 2pi); reduced-domain U[-a, a], a = s*pi/sqrt(2D)
with s = 1 (main) and s in REDUCED_SCALES_ABLATION (ablation); identity blocks.
The teacher stream is shared by all initializations (same targets per n).

results/init_variance.csv          one row per (n, init, scale, cost)
results/init_variance_fits.csv     decay rate b over n = 4..14 with bootstrap CI
results/init_variance_ratios.csv   Var[init] / Var[uniform] with bootstrap CI (H3 falsifier: n >= 10)
results/init_variance_layers.csv   variance averaged within each rotation layer
"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bp import SEED
from bp.ansatz import depth_log, ib_hea, layer_of_param, make_init, observable, trivial_params
from bp.config import INIT_KINDS, INIT_NS, N_BOOT, REDUCED_SCALES_ABLATION, VAR_K_FIXED, var_draws
from bp.data import results_dir
from bp.state_task import teacher_state
from bp.variance import bootstrap, fit_decay, variance_stats

OUT = results_dir(2)
CHUNK = 50
MAX_WORKERS = 14
CELLS = [(k, 1.0) for k in INIT_KINDS] + [("reduced", s) for s in REDUCED_SCALES_ABLATION]


def job(args):
    n, init, scale = args
    D = depth_log(n)
    circ = ib_hea(n, D)
    m = var_draws(n)
    rng_t = np.random.default_rng([SEED, n, 1])                       # shared targets
    rng_i = np.random.default_rng([SEED, n, 2, INIT_KINDS.index(init), int(scale * 100)])
    grads = {c: np.empty((m, circ.n_params)) for c in ("global", "local")}
    t0 = time.time()
    for s in range(0, m, CHUNK):
        b = min(CHUNK, m - s)
        psi0 = teacher_state(circ, rng_t, b)
        theta = make_init(init, rng_i, (b, circ.n_params), n, D, scale=scale * np.pi)
        for c in grads:
            grads[c][s:s + b] = -circ.expval_and_grad(theta, psi0, observable(n, c))[1]
    keep = np.setdiff1d(np.arange(circ.n_params), trivial_params(circ))
    layers = layer_of_param(n, circ.n_params)
    rows, boots, lay = [], {}, []
    for c, g in grads.items():
        v_k, v_mean = variance_stats(g, n, 2 * D, VAR_K_FIXED, keep)
        bs = bootstrap(g, n, 2 * D, np.random.default_rng([SEED, n, 7]), N_BOOT, VAR_K_FIXED, keep)
        boots[c] = bs
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        rows.append(dict(n=n, D=D, L=2 * D, init=init, scale=scale, cost=c, draws=m,
                         params=circ.n_params, var_k=v_k, var_k_lo=lo[0], var_k_hi=hi[0],
                         var_mean=v_mean, var_mean_lo=lo[1], var_mean_hi=hi[1],
                         mean_abs_grad=float(np.abs(g[:, keep]).mean())))
        v = g.var(axis=0, ddof=1)
        for l in range(2 * D):
            idx = np.intersect1d(np.where(layers == l)[0], keep)
            lay.append(dict(n=n, init=init, scale=scale, cost=c, layer=l, var_mean=float(v[idx].mean())))
    print(f"n={n:>2} {init:>8} s={scale:<4} Var_G={rows[0]['var_k']:.2e} Var_L={rows[1]['var_k']:.2e} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return (n, init, scale), rows, boots, lay


def main():
    jobs = sorted([(n, k, s) for n in INIT_NS for k, s in CELLS], key=lambda j: -j[0])
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        res = list(ex.map(job, jobs))
    rows = [r for _, rr, _, _ in res for r in rr]
    lay = [r for _, _, _, ll in res for r in ll]
    boots = {(key, c): b for key, _, bb, _ in res for c, b in bb.items()}
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "init_variance.csv", index=False)
    pd.DataFrame(lay).to_csv(OUT / "init_variance_layers.csv", index=False)

    fits = []
    ns = [n for n in INIT_NS if n >= 4]
    for init, scale in CELLS:
        for c in ("global", "local"):
            sub = df[(df.init == init) & (df.scale == scale) & (df.cost == c)].set_index("n").loc[ns]
            for j, stat in enumerate(("var_k", "var_mean")):
                _, b = fit_decay(ns, sub[stat])
                bb = np.array([fit_decay(ns, [boots[((n, init, scale), c)][i, j] for n in ns])[1]
                               for i in range(N_BOOT)])
                fits.append(dict(init=init, scale=scale, cost=c, stat=stat, b=b,
                                 b_lo=np.percentile(bb, 2.5), b_hi=np.percentile(bb, 97.5)))
    pd.DataFrame(fits).to_csv(OUT / "init_variance_fits.csv", index=False)

    ratios = []
    for n in INIT_NS:
        for init, scale in CELLS:
            if init == "uniform":
                continue
            for c in ("global", "local"):
                for j, stat in enumerate(("var_k", "var_mean")):
                    num, den = boots[((n, init, scale), c)][:, j], boots[((n, "uniform", 1.0), c)][:, j]
                    point = (df[(df.n == n) & (df.init == init) & (df.scale == scale) & (df.cost == c)][stat].iloc[0]
                             / df[(df.n == n) & (df.init == "uniform") & (df.cost == c)][stat].iloc[0])
                    r = num / den
                    ratios.append(dict(n=n, init=init, scale=scale, cost=c, stat=stat, ratio=point,
                                       ratio_lo=np.percentile(r, 2.5), ratio_hi=np.percentile(r, 97.5)))
    pd.DataFrame(ratios).to_csv(OUT / "init_variance_ratios.csv", index=False)
    print(pd.DataFrame(fits).round(3).to_string(index=False))
    rr = pd.DataFrame(ratios)
    print(rr[(rr.n >= 10) & (rr.stat == "var_k")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
