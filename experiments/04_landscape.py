"""2D slices of the cost landscape for the 3D visualisation (figures/landscape_3d.html).

A slice is C(theta_c + a d1 + b d2) on a grid of (a, b), where d1, d2 are random
orthogonal directions in the full 2nL-dimensional parameter space, scaled so
that a unit step moves every parameter by ~1 rad (RMS). Two centres:
  * "solution": theta_c = 0, where V = identity and both costs are exactly 0;
  * "random":   theta_c ~ U[0, 2 pi], a typical starting point for training.
The same directions and centres are used for the global and the local cost.
A 2D slice of a high-dimensional landscape is an illustration, not evidence;
the evidence is the variance sweep (01_variance_sweep.py).
"""
import json

import numpy as np

from bp import SEED
from bp.ansatz import depth_log, hea, observable
from bp.data import ROOT
from bp.sim import zero_state

NS = [2, 4, 6, 8, 10, 12]
GRID = 49
SPAN = 1.6           # a, b in [-SPAN, SPAN]


def directions(rng, P):
    d = rng.normal(size=(2, P))
    d[1] -= d[0] * (d[0] @ d[1]) / (d[0] @ d[0])
    return d / np.linalg.norm(d, axis=1, keepdims=True) * np.sqrt(P)


def slice_costs(n, L, centre, d, chunk=400):
    circ = hea(n, L)
    a = np.linspace(-SPAN, SPAN, GRID)
    A, B = np.meshgrid(a, a, indexing="ij")
    thetas = centre + A.reshape(-1, 1) * d[0] + B.reshape(-1, 1) * d[1]
    obs = {c: observable(n, c) for c in ("global", "local")}
    out = {c: np.empty(len(thetas)) for c in obs}
    for s in range(0, len(thetas), chunk):
        th = thetas[s:s + chunk]
        probs = circ.probabilities(th, zero_state(n, len(th)))
        for c, o in obs.items():
            out[c][s:s + chunk] = 1 - probs @ o
    return {c: np.round(v.reshape(GRID, GRID), 4).tolist() for c, v in out.items()}


data = dict(ns=NS, Ls=[depth_log(n) for n in NS], axis=np.linspace(-SPAN, SPAN, GRID).round(4).tolist(),
            solution={}, random={})
for n in NS:
    L = depth_log(n)
    P = 2 * n * L
    rng = np.random.default_rng([SEED, n, 99])
    d = directions(rng, P)
    data["solution"][n] = slice_costs(n, L, np.zeros(P), d)
    data["random"][n] = slice_costs(n, L, rng.uniform(0, 2 * np.pi, P), d)
    s = data["solution"][n]
    print(f"n={n:>2} L={L} P={P:>3}  solution slice: mean C_G={np.mean(s['global']):.3f} "
          f"mean C_L={np.mean(s['local']):.3f}", flush=True)

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "landscape.json").write_text(json.dumps(data))
