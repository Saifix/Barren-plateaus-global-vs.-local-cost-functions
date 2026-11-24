"""figures/variance_scaling.{pdf,png} and figures/analytic_check.{pdf,png}."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bp import SEED
from bp.ansatz import global_obs, local_obs, product_rx
from bp.data import results_dir
from bp.sim import zero_state
from bp.style import COLORS, IEEE_COL, IEEE_FULL, LABELS, save, use_style
from bp.variance import analytic_product_var

use_style()

# ---------------------------------------------------------------- scaling
df = pd.read_csv(results_dir(1) / "variance_sweep.csv")
fits = pd.read_csv(results_dir(1) / "variance_decay_fits.csv")
titles = {"1": "$L = 1$", "log": r"$L = \lceil \log_2 n \rceil$", "n": "$L = n$"}
fig, axes = plt.subplots(1, 3, figsize=(IEEE_FULL, 2.4), sharey=True)
for ax, depth in zip(axes, ["1", "log", "n"]):
    for j, cost in enumerate(("global", "local")):
        s = df[(df.depth == depth) & (df.cost == cost)].sort_values("n")
        fit = fits[(fits.depth == depth) & (fits.cost == cost) & (fits.stat == "var_k")].iloc[0]
        ax.fill_between(s.n, s.var_k_lo, s.var_k_hi, color=COLORS[cost], alpha=0.18, lw=0)
        ax.plot(s.n, s.var_k, "o-", color=COLORS[cost], label=LABELS[cost])
        ax.text(0.97, 0.95 - 0.1 * j,
                f"$b_{{{cost[0].upper()}}} = {fit.b:.2f}$ [{fit.b_lo:.2f}, {fit.b_hi:.2f}]",
                transform=ax.transAxes, ha="right", va="top", color=COLORS[cost], fontsize=7)
    ax.set_yscale("log")
    ax.set_title(titles[depth])
    ax.set_xlabel("qubits $n$")
    ax.set_xticks(sorted(df.n.unique()))
axes[0].set_ylabel(r"Var$[\partial C / \partial \theta_1]$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=0.6)
save(fig, "variance_scaling", 1)

# ---------------------------------------------------------- analytic check
rng = np.random.default_rng(SEED)
ns, m, chunk = np.arange(1, 13), 20_000, 2_000
est = {c: [] for c in ("global", "local")}
for n in ns:
    circ = product_rx(n)
    g = {c: [] for c in est}
    for s in range(0, m, chunk):
        th = rng.uniform(0, 2 * np.pi, (chunk, n))
        for c, obs in (("global", global_obs(n)), ("local", local_obs(n))):
            g[c].append(circ.expval_and_grad(th, zero_state(n, chunk), obs)[1][:, 0])
    for c in est:
        x = np.concatenate(g[c])
        v = x.var(ddof=1)
        se = np.sqrt((np.mean((x - x.mean()) ** 4) - v ** 2) / m)
        est[c].append((v, se))

fig, ax = plt.subplots(figsize=(IEEE_COL, 2.2))
nn = np.linspace(1, 12, 200)
for c in est:
    v, se = np.array(est[c]).T
    ax.plot(nn, [analytic_product_var(x, c) for x in nn], color=COLORS[c], lw=1, alpha=0.8)
    ax.errorbar(ns, v, yerr=2 * se, fmt="o", color=COLORS[c], ms=3, capsize=1.5, lw=0.8,
                label=f"{LABELS[c]}: simulated (±2 s.e.)")
ax.plot([], [], color="k", lw=1, label="closed form")
ax.set_yscale("log")
ax.set_xlabel("qubits $n$")
ax.set_ylabel(r"Var$[\partial C / \partial \theta_1]$")
ax.set_title("Product ansatz: simulator vs. closed form")
ax.legend(loc="lower left")
save(fig, "analytic_check", 1)
print("saved figures/variance_scaling and figures/analytic_check")
