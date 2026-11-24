"""figures/init_variance.{pdf,png}: H3 part 1, initial gradient variance by initialization."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from bp.data import results_dir
from bp.style import COLORS, IEEE_FULL, INIT_LABELS, INIT_STYLE, LABELS, save, use_style

use_style()
df = pd.read_csv(results_dir(2) / "init_variance.csv")
fits = pd.read_csv(results_dir(2) / "init_variance_fits.csv")
ratios = pd.read_csv(results_dir(2) / "init_variance_ratios.csv")

fig, axes = plt.subplots(1, 3, figsize=(IEEE_FULL, 2.4))
for ax, cost in zip(axes[:2], ("global", "local")):
    for init in ("uniform", "reduced", "identity"):
        s = df[(df.cost == cost) & (df.init == init) & (df.scale == 1.0)].sort_values("n")
        st = INIT_STYLE[init]
        ax.fill_between(s.n, s.var_k_lo, s.var_k_hi, color=COLORS[cost], alpha=0.10, lw=0)
        ax.plot(s.n, s.var_k, color=COLORS[cost], ls=st["ls"], marker=st["marker"], mfc="white")
    txt = []
    for init in ("uniform", "reduced", "identity"):
        f = fits[(fits.init == init) & (fits.scale == 1.0) & (fits.cost == cost)
                 & (fits.stat == "var_k")].iloc[0]
        txt.append(f"{init[:3]}: $b$ = {f.b:.2f} [{f.b_lo:.2f}, {f.b_hi:.2f}]")
    corner = dict(x=0.03, y=0.03, ha="left", va="bottom") if cost == "global" else \
        dict(x=0.97, y=0.97, ha="right", va="top")
    ax.text(corner["x"], corner["y"], "\n".join(txt), transform=ax.transAxes, fontsize=6.3,
            color=COLORS[cost], ha=corner["ha"], va=corner["va"])
    ax.set_yscale("log")
    ax.set_xlabel("qubits $n$")
    ax.set_xticks(sorted(df.n.unique()))
    ax.set_title(LABELS[cost])
axes[0].set_ylabel(r"initial Var$[\partial C/\partial\theta_1]$")

# ratio to uniform at every n (H3 falsifier line at 10x)
ax = axes[2]
for cost in ("global", "local"):
    for init in ("reduced", "identity"):
        r = ratios[(ratios.cost == cost) & (ratios.init == init) & (ratios.scale == 1.0)
                   & (ratios.stat == "var_k")].sort_values("n")
        st = INIT_STYLE[init]
        ax.errorbar(r.n + (0.12 if cost == "local" else -0.12), r.ratio,
                    yerr=[r.ratio - r.ratio_lo, r.ratio_hi - r.ratio], color=COLORS[cost],
                    ls=st["ls"], marker=st["marker"], mfc="white", capsize=1.5, lw=1)
ax.axhline(1, color="k", lw=0.6)
ax.axhline(10, color="k", lw=0.8, ls=":")
ax.text(2.1, 10.5, "H3 threshold (10$\\times$)", fontsize=6.5, va="bottom")
ax.axvspan(9.5, 14.5, color="0.5", alpha=0.08, lw=0)
ax.set_yscale("log")
ax.set_xlabel("qubits $n$")
ax.set_xticks(sorted(df.n.unique()))
ax.set_title("ratio to uniform init")
handles = [Line2D([], [], color="k", ls=INIT_STYLE[i]["ls"], marker=INIT_STYLE[i]["marker"],
                  mfc="white", label=INIT_LABELS[i]) for i in ("uniform", "reduced", "identity")]
handles += [Line2D([], [], color=COLORS[c], label=LABELS[c]) for c in ("global", "local")]
fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(w_pad=0.8)
save(fig, "init_variance", 2)

# Ablation: reduced-domain width, global and local, ratio to uniform
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for cost in ("global", "local"):
    for scale, mk in ((0.25, "v"), (0.5, "s"), (1.0, "o")):
        r = ratios[(ratios.cost == cost) & (ratios.init == "reduced") & (ratios.scale == scale)
                   & (ratios.stat == "var_k")].sort_values("n")
        ax.plot(r.n, r.ratio, color=COLORS[cost], marker=mk, mfc="white", lw=1,
                label=f"{cost}, $a = {scale:g}\\pi/\\sqrt{{L}}$")
ax.axhline(1, color="k", lw=0.6)
ax.axhline(10, color="k", lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_xlabel("qubits $n$")
ax.set_ylabel("Var[reduced] / Var[uniform]")
ax.set_title("Ablation: reduced-domain width")
ax.legend(ncol=2, fontsize=6)
save(fig, "init_reduced_width", 2)
print("saved figures/init_variance, figures/init_reduced_width")
