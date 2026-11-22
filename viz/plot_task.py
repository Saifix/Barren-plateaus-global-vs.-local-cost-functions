"""figures/task_training.{pdf,png}: H3 part 2, training on the teacher-state task."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from bp.config import TASK_ITERS, TASK_SHOTS, TASK_TARGET
from bp.data import ROOT
from bp.style import COLORS, IEEE_FULL, INIT_LABELS, INIT_STYLE, LABELS, save, use_style

use_style()
c = pd.read_csv(ROOT / "results" / "task_curves.csv")
fin = pd.read_csv(ROOT / "results" / "task_final.csv")
SHOW = {"shots": [6, 8, 10], "exact": [8, 10, 12]}    # shot runs stop at n = 10

fig, axes = plt.subplots(2, 4, figsize=(IEEE_FULL, 3.9),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
for row, mode in enumerate(("shots", "exact")):
    for j, n in enumerate(SHOW[mode]):
        ax = axes[row, j]
        for cost in ("global", "local"):
            for init in ("uniform", "reduced", "identity"):
                s = c[(c.n == n) & (c.cost == cost) & (c.init == init) & (c["mode"] == mode)]
                med = s.groupby("iter").C_G.median()
                ax.plot(med.index, med.values, color=COLORS[cost], ls=INIT_STYLE[init]["ls"], lw=1.2)
        ax.axhline(TASK_TARGET, color="k", lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(3e-3, 1.3)
        ax.set_title(f"$n={n}$, " + (f"{TASK_SHOTS}-shot gradients" if mode == "shots"
                                     else "exact gradients"), fontsize=7.5)
        if row == 1:
            ax.set_xlabel("iteration")
        if j == 0:
            ax.set_ylabel(r"median $C_G$ (exact)")
    # right column: fraction of seeds reaching the target, vs n
    ax = axes[row, -1]
    for cost in ("global", "local"):
        for init in ("uniform", "reduced", "identity"):
            s = fin[(fin.cost == cost) & (fin.init == init) & (fin["mode"] == mode)]
            r = s.groupby("n").reached.mean()
            ax.plot(r.index + {"uniform": -0.15, "reduced": 0, "identity": 0.15}[init], r.values,
                    color=COLORS[cost], ls=INIT_STYLE[init]["ls"],
                    marker=INIT_STYLE[init]["marker"], mfc="white", lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sorted(fin.n.unique()))
    ax.set_ylabel(f"fraction reaching $C_G\\leq{TASK_TARGET}$")
    ax.set_title(f"success in {TASK_ITERS} iterations", fontsize=7.5)
    if row == 1:
        ax.set_xlabel("qubits $n$")
handles = [Line2D([], [], color=COLORS[k], label=LABELS[k] + " (training loss)") for k in ("global", "local")]
handles += [Line2D([], [], color="k", ls=INIT_STYLE[i]["ls"], marker=INIT_STYLE[i]["marker"],
                   mfc="white", label=INIT_LABELS[i])
            for i in ("uniform", "reduced", "identity")]
fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05), fontsize=6.5)
fig.tight_layout(h_pad=0.6, w_pad=0.5)
save(fig, "task_training")
print("saved figures/task_training")
