"""figures/noise_variance.{pdf,png} (density-matrix study) and
figures/fake_backend.{pdf,png} (FakeFez noise model)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from bp.config import HW_SHOTS, NOISE_DEPTH_N, NOISE_P2
from bp.data import results_dir
from bp.style import COLORS, IEEE_COL, IEEE_FULL, LABELS, NOISE_LABELS, NOISE_STYLE, save, use_style

use_style()
df = pd.read_csv(results_dir(2) / "noise_variance.csv")
lay = pd.read_csv(results_dir(2) / "noise_variance_layers.csv")
MODELS = ("noiseless", "depolarizing", "amplitude_damping")


def line(ax, x, y, cost, model, lo=None, hi=None):
    st = NOISE_STYLE[model]
    if lo is not None:
        ax.fill_between(x, lo, hi, color=COLORS[cost], alpha=0.08, lw=0)
    ax.plot(x, y, color=COLORS[cost], ls=st["ls"], marker=st["marker"], mfc="white", lw=1.1)


fig, axes = plt.subplots(1, 4, figsize=(IEEE_FULL, 2.3))
ax = axes[0]
for cost in ("global", "local"):
    for m in MODELS:
        s = df[(df.study == "n") & (df.cost == cost) & (df.model == m)].sort_values("n")
        line(ax, s.n, s.var_k, cost, m, s.var_k_lo, s.var_k_hi)
ax.set_xlabel("qubits $n$  ($L=\\lceil\\log_2 n\\rceil$)")
ax.set_ylabel(r"Var$[\partial C/\partial\theta_1]$")
ax.set_title(f"(a) vs. $n$, $p_2={NOISE_P2}$")
ax.set_xticks(sorted(df[df.study == "n"].n.unique()))

ax = axes[1]
for cost in ("global", "local"):
    for m in MODELS:
        s = df[(df.study == "depth") & (df.cost == cost) & (df.model == m)].sort_values("L")
        line(ax, s.L, s.var_mean, cost, m, s.var_mean_lo, s.var_mean_hi)
ax.set_xlabel("layers $L$")
ax.set_ylabel(r"Var averaged over $k$")
ax.set_title(f"(b) vs. depth, $n={NOISE_DEPTH_N}$")

ax = axes[2]
Lmax = lay[lay.study == "depth"].L.max()
for cost in ("global", "local"):
    for m in MODELS:
        s = lay[(lay.study == "depth") & (lay.L == Lmax) & (lay.cost == cost) & (lay.model == m)]
        line(ax, s.layer + 1, s.var_mean, cost, m)
ax.set_xlabel("layer of the parameter")
ax.set_ylabel("Var averaged within layer")
ax.set_title(f"(c) per layer, $n={NOISE_DEPTH_N}$, $L={Lmax}$")

ax = axes[3]
ref = df[(df.study == "p") & (df.model == "noiseless")]
for cost in ("global", "local"):
    v0 = ref[ref.cost == cost].var_mean.iloc[0]
    for m in MODELS[1:]:
        s = df[(df.study == "p") & (df.cost == cost) & (df.model == m)].sort_values("p2")
        line(ax, s.p2, s.var_mean / v0, cost, m)
ax.axhline(1, color="k", lw=0.6)
ax.set_xscale("log")
ax.set_xlabel("two-qubit noise strength $p_2$")
ax.set_ylabel("Var / noiseless Var")
ax.set_title(f"(d) strength, $n={NOISE_DEPTH_N}$, $L=6$")
for a in axes:
    a.set_yscale("log")
handles = [Line2D([], [], color="k", ls=NOISE_STYLE[m]["ls"], marker=NOISE_STYLE[m]["marker"],
                  mfc="white", label=NOISE_LABELS[m]) for m in MODELS]
handles += [Line2D([], [], color=COLORS[c], label=LABELS[c]) for c in ("global", "local")]
fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(w_pad=0.4)
save(fig, "noise_variance", 2)

# ------------------------------------------------------------ fake backend
fb = pd.read_csv(results_dir(2) / "fake_backend.csv")
fig, axes = plt.subplots(1, 2, figsize=(IEEE_FULL * 0.62, 2.2), sharey=True)
for ax, k in zip(axes, ("first", "last")):
    for cost in ("global", "local"):
        for model, style in (("noiseless", NOISE_STYLE["noiseless"]), ("fake", NOISE_STYLE["fake"])):
            s = fb[(fb.k == k) & (fb.cost == cost) & (fb.model == model)].sort_values("n")
            ax.errorbar(s.n, s["var"], yerr=[s["var"] - s.var_lo, s.var_hi - s["var"]],
                        color=COLORS[cost], ls=style["ls"], marker=style["marker"], mfc="white",
                        capsize=1.5, lw=1)
        s = fb[(fb.k == k) & (fb.cost == cost) & (fb.model == "fake")].sort_values("n")
        ax.plot(s.n, s.shot_floor, color=COLORS[cost], lw=0.8, alpha=0.6, ls=(0, (1, 1)))
    ax.set_yscale("log")
    ax.set_xticks(sorted(fb.n.unique()))
    ax.set_xlabel("qubits $n$")
    ax.set_title("$R_Y$ on qubit 1, " + ("first layer" if k == "first" else "last layer"))
axes[0].set_ylabel(r"Var$[\partial C/\partial\theta_k]$")
handles = [Line2D([], [], color="k", **{"ls": NOISE_STYLE["noiseless"]["ls"]}, marker="o", mfc="white",
                  label="noiseless"),
           Line2D([], [], color="k", ls=NOISE_STYLE["fake"]["ls"], marker="D", mfc="white",
                  label="FakeFez noise model"),
           Line2D([], [], color="k", ls=(0, (1, 1)), label=f"shot-noise floor ({HW_SHOTS} shots)")]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09), fontsize=6.5)
fig.tight_layout(w_pad=0.4)
save(fig, "fake_backend", 2)
print("saved figures/noise_variance, figures/fake_backend")
