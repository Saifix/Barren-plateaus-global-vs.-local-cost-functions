"""Gradient-variance estimation with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np

from .ansatz import hea, init_uniform, observable, structurally_zero_params
from .sim import zero_state


def sample_gradients(n, L, n_draws, rng, costs=("global", "local"), chunk=50):
    """Exact cost gradients at ``n_draws`` uniform random parameter vectors.

    The same parameter draws are used for every cost (paired comparison).
    Returns ``{cost: array (n_draws, P)}`` of dC/dtheta with C = 1 - <O>.
    """
    circ = hea(n, L)
    theta = init_uniform(rng, (n_draws, circ.n_params))
    out = {c: np.empty((n_draws, circ.n_params)) for c in costs}
    for s in range(0, n_draws, chunk):
        th = theta[s:s + chunk]
        psi0 = zero_state(n, len(th))
        for c in costs:
            _, g = circ.expval_and_grad(th, psi0, observable(n, c))
            out[c][s:s + chunk] = -g
    return out


def variance_stats(grads, n, L, k_fixed=0, keep=None):
    """(Var of dC/dtheta_k at fixed k, Var averaged over the non-trivial k).

    ``keep`` = indices to average over; default: all but the structurally zero
    angles of the standard HEA (pass ``trivial_params`` for other circuits)."""
    if keep is None:
        keep = np.setdiff1d(np.arange(grads.shape[1]), structurally_zero_params(n, L))
    v = grads.var(axis=0, ddof=1)
    return float(v[k_fixed]), float(v[keep].mean())


def bootstrap(grads, n, L, rng, n_boot=1000, k_fixed=0, keep=None):
    """Bootstrap over parameter draws: ``(n_boot, 2)`` array of (fixed-k, mean-k) variances."""
    m = grads.shape[0]
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        out[b] = variance_stats(grads[rng.integers(0, m, m)], n, L, k_fixed, keep)
    return out


def fit_decay(ns, variances):
    """Least-squares fit of log Var = a - b n. Returns (a, b)."""
    slope, intercept = np.polyfit(np.asarray(ns, float), np.log(variances), 1)
    return intercept, -slope


# Closed forms for the product ansatz  (x)_j R_X(theta_j) |0>, theta ~ U[0, 2pi]
def analytic_product_var(n, cost):
    return 0.125 * 0.375 ** (n - 1) if cost == "global" else 1.0 / (8 * n * n)
