"""The H3 training task: learn V(theta) that maps a fixed state back to |0...0>.

Target. A teacher angle vector theta* (uniform in [0, 2pi)) of the *same*
circuit prepares |psi0> = V(theta*)^dagger |0...0>. So the task is exactly
solvable (theta = theta* reaches C = 0), the identity initialization starts
far from the solution, and every (cost, init, gradient mode) cell with the
same seed sees the same target (a paired design).

Costs (the training loss is the cost itself):
  global  C_G = 1 - P(0^n)                       (= infidelity with |0...0>)
  local   C_L = 1 - (1/n) sum_j P(q_j = 0)
Success is measured on C_G for both: C_L = 0 iff C_G = 0, and
C_L <= C_G <= n C_L, so a low C_L alone does not mean a good state.

Gradient modes:
  exact  adjoint gradients
  shots  parameter-shift gradients, each of the 2P shifted circuits estimated
         from ``shots`` samples of its output distribution (what a device does)
"""
from __future__ import annotations

import numpy as np

from .ansatz import ib_hea, init_uniform, make_init, observable
from .qclassifier import Adam
from .sim import zero_state

INITS = ("uniform", "reduced", "identity")


def inverse_run(circ, theta, psi):
    """V(theta)^dagger |psi>: the gates in reverse order with negated angles."""
    theta = np.asarray(theta, float)
    state = np.array(psi, dtype=complex, copy=True)
    for op in reversed(circ.ops):
        state = circ._apply(state, op, theta, inverse=True)
    return state


def teacher_state(circ, rng, batch: int = 1):
    """``batch`` targets |psi0> = V(theta*)^dagger |0>, theta* ~ U[0, 2pi)."""
    theta_star = init_uniform(rng, (batch, circ.n_params))
    return inverse_run(circ, theta_star, zero_state(circ.n, batch))


def shot_gradient(circ, theta, psi0, shots, rng, obs, chunk=64):
    """Parameter-shift gradient of C = 1 - <O>, every circuit from ``shots`` samples.

    The shifted circuits are simulated in chunks so
    memory stays at ``chunk * 2**n`` amplitudes; multinomial draws are made in
    row order, so chunking does not change the random stream.
    """
    P = circ.n_params
    shifts = np.vstack([np.eye(P) * np.pi / 2, -np.eye(P) * np.pi / 2])
    est = np.empty(2 * P)
    for s in range(0, 2 * P, chunk):
        th = theta[None] + shifts[s:s + chunk]
        probs = circ.probabilities(th, np.repeat(psi0, len(th), axis=0))
        probs = np.clip(probs, 0, None)
        probs /= probs.sum(axis=1, keepdims=True)
        counts = rng.multinomial(shots, probs)
        est[s:s + chunk] = 1 - counts @ obs / shots
    return (est[:P] - est[P:]) / 2


def train_state(n: int, D: int, cost: str, init: str, seed: int, mode: str = "exact",
                shots: int = 1024, iters: int = 200, lr: float = 0.05, target: float = 0.1,
                blocks: int = 1):
    """Train ``ib_hea(n, D, blocks)`` on the teacher task. Returns a list of dicts
    (one per iteration, iteration 0 = initialization) with exact C_G, C_L and
    the exact and used gradient norms."""
    circ = ib_hea(n, D, blocks)
    psi0 = teacher_state(circ, np.random.default_rng([seed, 1]))
    theta = make_init(init, np.random.default_rng([seed, 2]), circ.n_params, n, D, blocks)
    shot_rng = np.random.default_rng([seed, 3, ("global", "local").index(cost)])
    obs = observable(n, cost)
    obs_g, obs_l = observable(n, "global"), observable(n, "local")
    opt, hist = Adam(lr), []
    for it in range(iters + 1):
        _, g = circ.expval_and_grad(theta, psi0, obs)
        g_exact = -g[0]
        p = circ.probabilities(theta, psi0)[0]
        row = dict(iter=it, C_G=float(1 - p @ obs_g), C_L=float(1 - p @ obs_l),
                   exact_grad_norm=float(np.linalg.norm(g_exact)))
        if it == iters:
            row["grad_norm"] = row["exact_grad_norm"]
            hist.append(row)
            break
        g_used = (g_exact if mode == "exact"
                  else shot_gradient(circ, theta, psi0, shots, shot_rng, obs))
        row["grad_norm"] = float(np.linalg.norm(g_used))
        hist.append(row)
        theta = opt.step(theta, g_used)
    return hist


def iters_to_target(hist, target=0.1, key="C_G"):
    """First iteration with ``key <= target``; None if never reached."""
    for row in hist:
        if row[key] <= target:
            return row["iter"]
    return None
