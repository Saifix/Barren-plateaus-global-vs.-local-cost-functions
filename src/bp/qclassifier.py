"""Variational quantum classifier: angle encoding -> HEA -> global or local readout.

Class score p(x) = P(class +1 | x):
  global readout  p = P(0^n)                                  = <O_G>
  local  readout  p = (1/n) sum_j P(q_j = 0) = (1 + <Z>avg)/2  = <O_L>
There is no classical post-processing and no trainable bias: the readout is
the circuit output itself, so global vs. local is the only difference.

Two gradient modes
------------------
shots=None  exact gradients (adjoint method).
shots=N     what a device would do: every circuit (p at theta, and at
            theta +- pi/2 e_k for the parameter-shift rule) is estimated from N
            multinomial samples of its exact output distribution, so the
            gradient carries realistic shot noise.

Why both matter: Adam rescales each coordinate by a running RMS of its
gradient, so an exponentially small but *exact* gradient still produces
lr-sized steps, and cross-entropy adds a 1/p factor on top. Exact training
therefore cannot show a barren plateau; it appears when the gradient signal
is small compared with its shot noise.

Two losses
----------
"bce"    binary cross-entropy. Under shots, P(0^n) ~ 2^-n is often estimated as
         0 counts, where dBCE/dp = -1/p diverges, so BCE is used with exact
         gradients only.
"brier"  mean squared error (p - t)^2, a proper scoring rule with bounded
         dL/dp = 2(p - t). Used for shot-based training and its exact control.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ansatz import hea, init_uniform, observable
from .metrics import EPS, scores
from .sim import angle_encode


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, theta, g):
        if self.m is None:
            self.m, self.v = np.zeros_like(g), np.zeros_like(g)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * mh / (np.sqrt(vh) + self.eps)


def dloss_dp(loss: str, p, t):
    """d(mean loss)/dp for each sample."""
    if loss == "bce":
        pc = np.clip(p, EPS, 1 - EPS)
        return (pc - t) / (pc * (1 - pc)) / len(t)
    if loss == "brier":
        return 2 * (p - t) / len(t)
    raise ValueError(loss)


def loss_value(loss: str, p, t):
    if loss == "bce":
        pc = np.clip(p, EPS, 1 - EPS)
        return float(-np.mean(t * np.log(pc) + (1 - t) * np.log(1 - pc)))
    return float(np.mean((p - t) ** 2))


@dataclass
class QuantumClassifier:
    n: int
    L: int
    readout: str = "local"          # "global" | "local"
    loss: str = "bce"               # "bce" | "brier"
    shots: int | None = None        # None = exact gradients
    lr: float = 0.05
    epochs: int = 60
    batch_size: int = 32
    seed: int = 0
    chunk_rows: int = 4096          # shot mode: circuits simulated at once
    theta: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.shots and self.loss == "bce":
            raise ValueError("BCE diverges under shot noise (p_hat = 0); use loss='brier'")
        self.circuit = hea(self.n, self.L)
        self.obs = observable(self.n, self.readout)
        self.n_params = self.circuit.n_params

    # ------------------------------------------------------------------
    def predict_proba(self, X, theta=None):
        theta = self.theta if theta is None else theta
        return self.circuit.probabilities(theta, angle_encode(X)) @ self.obs

    def exact_grad(self, theta, X, t):
        """Exact loss and gradient (adjoint method)."""
        p, g = self.circuit.expval_and_grad(theta, angle_encode(X), self.obs,
                                            weights=lambda p: dloss_dp(self.loss, p, t))
        return loss_value(self.loss, p, t), g.sum(axis=0)

    def sampled_readout(self, probs, rng):
        """Estimate the readout of each row of ``probs`` from ``self.shots`` samples."""
        probs = np.clip(probs, 0, None)
        probs /= probs.sum(axis=1, keepdims=True)
        counts = rng.multinomial(self.shots, probs)
        return counts @ self.obs / self.shots

    def shot_grad(self, theta, X, t, rng):
        """Parameter-shift gradient with every circuit estimated from ``shots`` samples.

        p at theta and p at theta +- pi/2 e_k are independent estimates, so the
        product dL/dp * dp/dtheta is unbiased for the Brier loss.
        """
        P, B = self.n_params, len(X)
        shifts = np.vstack([np.zeros(P), np.eye(P) * np.pi / 2, -np.eye(P) * np.pi / 2])
        thetas = np.repeat(theta[None] + shifts, B, axis=0)            # (2P+1)*B rows
        psi_x = angle_encode(X)
        # Simulate in row chunks to bound memory; multinomial draws are made row
        # by row, so the random stream (and the result) does not depend on chunk.
        rows = len(thetas)
        step = max(B, (self.chunk_rows // B) * B)
        p_hat = np.empty(rows)
        for s in range(0, rows, step):
            th = thetas[s:s + step]
            psi0 = np.tile(psi_x, (len(th) // B, 1))
            p_hat[s:s + step] = self.sampled_readout(self.circuit.probabilities(th, psi0), rng)
        p_hat = p_hat.reshape(2 * P + 1, B)
        dp = (p_hat[1:P + 1] - p_hat[P + 1:]) / 2                       # (P, B)
        g = dp @ dloss_dp(self.loss, p_hat[0], t)
        return loss_value(self.loss, p_hat[0], t), g

    # ------------------------------------------------------------------
    def fit(self, X, y, X_test=None, y_test=None, log_every=1):
        """Returns the training history (one dict per logged epoch, epoch 0 = init).

        Logged per epoch: the norm of the gradient actually used for the update
        (``grad_norm``; shot-noisy when shots is set) and of the exact gradient
        at the same parameters (``exact_grad_norm``), plus exact train/test metrics.
        """
        rng = np.random.default_rng(self.seed)
        shot_rng = np.random.default_rng([self.seed, 7])
        theta = init_uniform(rng, self.n_params)
        t_all = (y == 1).astype(float)
        opt, history = Adam(self.lr), []

        def grad(th, idx):
            if self.shots:
                return self.shot_grad(th, X[idx], t_all[idx], shot_rng)[1]
            return self.exact_grad(th, X[idx], t_all[idx])[1]

        def log(epoch, used, exact):
            row = dict(epoch=epoch, grad_norm=used, exact_grad_norm=exact)
            row.update({f"train_{k}": v for k, v in scores(y, self.predict_proba(X, theta)).items()})
            if X_test is not None:
                row.update({f"test_{k}": v for k, v in
                            scores(y_test, self.predict_proba(X_test, theta)).items()})
            history.append(row)

        # epoch 0: full training set for exact gradients; one batch under shots
        # (drawn from shot_rng so the main stream -- and every exact run -- is unchanged)
        idx0 = (shot_rng.permutation(len(X))[:self.batch_size] if self.shots
                else np.arange(len(X)))
        g0 = float(np.linalg.norm(self.exact_grad(theta, X[idx0], t_all[idx0])[1]))
        log(0, float(np.linalg.norm(grad(theta, idx0))) if self.shots else g0, g0)
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(X))
            used, exact = [], []
            for s in range(0, len(X), self.batch_size):
                idx = order[s:s + self.batch_size]
                g = grad(theta, idx)
                used.append(np.linalg.norm(g))
                if self.shots:
                    exact.append(np.linalg.norm(self.exact_grad(theta, X[idx], t_all[idx])[1]))
                theta = opt.step(theta, g)
            if epoch % log_every == 0 or epoch == self.epochs:
                log(epoch, float(np.mean(used)), float(np.mean(exact or used)))
        self.theta = theta
        return history
