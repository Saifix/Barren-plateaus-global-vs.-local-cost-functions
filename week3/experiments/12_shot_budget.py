"""Q1: how many shots does each cost need before its gradient is measurable?

HEA, |0...0>, uniform random angles, L = ceil(log2 n), the H1 parameter
theta_1 (R_Y on qubit 0, first layer), n = 2..14, the Week 1 draw counts.
For every draw, the two parameter-shifted circuits theta_1 +- pi/2 are
simulated exactly. From their output distributions:
  g          = (m+ - m-) / 2         the exact gradient (parameter shift)
  s2+, s2-   single-shot variance of the cost's estimator: m(1 - m) for the
             global cost (a Bernoulli outcome), Var_x[O_L(x)] for the local one
  shot noise of an N-shot estimate of g:  (s2+ + s2-) / (4 N)
Signal = Var over draws of g. The shot budget for a signal-to-noise ratio of 1 is
  N* = E[s2+ + s2-] / (4 Var[g])   (shots per circuit).
The usual worst-case view takes s2 = 1/4 for every circuit (N*_worst =
1 / (8 Var[g])); for the global cost s2 ~ P(0^n) ~ 2^-n is far smaller, which
is why Week 2's global-cost training still worked at n = 10 with 1024 shots.
SNR at each budget in SHOT_GRID is reported, with 95% bootstrap intervals.

results/shot_budget.csv
"""
import numpy as np
import pandas as pd

from bp import SEED
from bp.ansatz import depth_for, hea, init_uniform, observable
from bp.config import N_BOOT, SHOT_GRID, SHOT_VAR_L, VAR_K_FIXED, VAR_NS, var_draws
from bp.data import results_dir
from bp.sim import zero_state

CHUNK = 250


def shifted_moments(circ, theta, obs, k):
    """Mean and single-shot variance of the observable at theta_k +- pi/2."""
    out = []
    for s in (+1, -1):
        th = theta.copy()
        th[:, k] += s * np.pi / 2
        p = circ.probabilities(th, zero_state(circ.n, len(th)))
        m = p @ obs
        out.append((m, np.clip(p @ obs ** 2 - m ** 2, 0, None)))
    return out


def main():
    rows = []
    for n in VAR_NS:
        L = depth_for(n, SHOT_VAR_L)
        circ = hea(n, L)
        D = var_draws(n)
        theta = init_uniform(np.random.default_rng([SEED, n, 12]), (D, circ.n_params))
        for cost in ("global", "local"):
            obs = observable(n, cost)
            g, s2 = np.empty(D), np.empty(D)
            for a in range(0, D, CHUNK):
                (mp, vp), (mm, vm) = shifted_moments(circ, theta[a:a + CHUNK], obs, VAR_K_FIXED)
                g[a:a + CHUNK] = (mp - mm) / 2
                s2[a:a + CHUNK] = vp + vm
            rng = np.random.default_rng([SEED, n, 13])
            idx = rng.integers(0, D, size=(N_BOOT, D))
            var_b = g[idx].var(axis=1, ddof=1)
            nstar_b = s2[idx].mean(axis=1) / (4 * var_b)
            var_g, nstar = g.var(ddof=1), s2.mean() / (4 * g.var(ddof=1))
            row = dict(n=n, L=L, cost=cost, draws=D, var=var_g,
                       var_lo=np.percentile(var_b, 2.5), var_hi=np.percentile(var_b, 97.5),
                       single_shot_var=s2.mean() / 2,
                       n_star=nstar, n_star_lo=np.percentile(nstar_b, 2.5),
                       n_star_hi=np.percentile(nstar_b, 97.5),
                       n_star_worst=1 / (8 * var_g))
            for N in SHOT_GRID:
                snr_b = 4 * N * var_b / s2[idx].mean(axis=1)
                row[f"snr_{N}"] = 4 * N * var_g / s2.mean()
                row[f"snr_{N}_lo"], row[f"snr_{N}_hi"] = np.percentile(snr_b, [2.5, 97.5])
            rows.append(row)
            print(f"n={n:>2} L={L} {cost:>6}: Var={var_g:.2e}  single-shot var={s2.mean() / 2:.2e}  "
                  f"N*={nstar:.3g} [{row['n_star_lo']:.3g}, {row['n_star_hi']:.3g}]  "
                  f"(worst-case view {row['n_star_worst']:.3g})", flush=True)
    pd.DataFrame(rows).to_csv(results_dir(3) / "shot_budget.csv", index=False)


if __name__ == "__main__":
    main()
