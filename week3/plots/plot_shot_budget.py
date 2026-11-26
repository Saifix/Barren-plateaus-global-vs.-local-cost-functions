"""figures/shot_budget.{pdf,png}: Q1, the shot budget as a hyperparameter.
(a) shots per circuit needed for gradient SNR = 1, vs n, against the worst-case
estimate; (b, c) teacher-task success at each budget (needs experiment 13)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from bp.config import SHOT_GRID
from bp.data import results_dir
from bp.style import COLORS, IEEE_FULL, LABELS, save, use_style

use_style()
R = results_dir(3)
sb = pd.read_csv(R / "shot_budget.csv")
tr_path = R / "shot_training_final.csv"
tr = pd.read_csv(tr_path) if tr_path.exists() else None

fig, axes = plt.subplots(1, 3, figsize=(IEEE_FULL, 2.3), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axes[0]
for cost in ("global", "local"):
    d = sb[sb.cost == cost].sort_values("n")
    ax.fill_between(d.n, d.n_star_lo, d.n_star_hi, color=COLORS[cost], alpha=0.15, lw=0)
    ax.plot(d.n, d.n_star, color=COLORS[cost], marker="o", mfc="white", lw=1.3)
    ax.plot(d.n, d.n_star_worst, color=COLORS[cost], ls=":", lw=1)
    fit = d[d.n >= 4]
    b = np.polyfit(fit.n, np.log(fit.n_star), 1)[0]
    ax.text(fit.n.iloc[-1] + 0.3, fit.n_star.iloc[-1], f"×{np.exp(2 * b):.1f}\nper 2 qubits",
            color=COLORS[cost], fontsize=5.8, va="center")
for N in SHOT_GRID:
    ax.axhline(N, color="0.6", lw=0.5, ls="--")
    # on-line labels; 5000 and 10000 are only 0.3 decades apart, so stagger them in x
    ax.text(4.6 if N == 5000 else 2.05, N, f"{N}", fontsize=5.5, color="0.45", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4))
ax.set_yscale("log")
ax.set_xlim(1.6, 16.8)
ax.set_xticks(sorted(sb.n.unique()))
ax.set_xlabel("qubits $n$  ($L=\\lceil\\log_2 n\\rceil$)")
ax.set_ylabel("shots per circuit for SNR $=1$")
ax.set_title("(a) shot budget $N^*$ for $\\partial_1 C$")
ax.legend(handles=[Line2D([], [], color="k", marker="o", mfc="white", label="exact $N^*$"),
                   Line2D([], [], color="k", ls=":", label="worst case ($\\sigma^2=1/4$)")],
          fontsize=5.8, loc="upper left")

markers = {6: "o", 8: "s", 10: "^", 12: "D"}
for ax, cost in zip(axes[1:], ("global", "local")):
    if tr is None:
        ax.text(0.5, 0.5, "training runs pending", ha="center", transform=ax.transAxes)
        continue
    for n, d in tr[tr.cost == cost].groupby("n"):
        s = d.groupby("shots").reached.mean()
        ax.plot(s.index, s.values, color=COLORS[cost], marker=markers.get(n, "o"), mfc="white", lw=1,
                alpha=0.45 + 0.55 * (n - 6) / 6)
    ax.set_xscale("log")
    ax.set_xticks(SHOT_GRID)
    ax.set_xticklabels([str(N) for N in SHOT_GRID], fontsize=6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("shots per circuit")
    ax.set_title(f"({'b' if cost == 'global' else 'c'}) teacher task, {LABELS[cost].split(' $')[0].lower()}")
axes[1].set_ylabel("seeds reaching $C_G\\leq0.1$")
axes[2].legend(handles=[Line2D([], [], color="0.3", marker=m, mfc="white", ls="-", label=f"$n={n}$")
                        for n, m in markers.items()], fontsize=5.8, loc="lower right")
fig.tight_layout(w_pad=0.6)
save(fig, "shot_budget", 3)
print("saved week3/figures/shot_budget")
