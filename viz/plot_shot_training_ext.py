"""figures/shot_training_ext.{pdf,png}: H4 at n = 10 (Run C) and with fewer shots (Run D).

Run D's 1024-shot and exact references are Run B's n = 8 runs (same folds,
seeds and settings), read from results/shot_curves.csv.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from bp.config import RUN_C_NS, RUN_D_N, RUN_D_SHOTS, SHOTS_TRAIN
from bp.data import ROOT
from bp.style import COLORS, IEEE_FULL, LABELS, save, use_style

use_style()
ext = pd.read_csv(ROOT / "results" / "shot_ext_curves.csv")
b = pd.read_csv(ROOT / "results" / "shot_curves.csv")
b = b[b.n == RUN_D_N].assign(shots_train=lambda d: d["mode"].map({"shots": SHOTS_TRAIN, "exact": 0}))
allc = pd.concat([ext, b.assign(run="B")], ignore_index=True)

panels = [(f"Run C: $n={n}$", allc[(allc.run == "C") & (allc.n == n)]) for n in RUN_C_NS]
panels += [(f"Run D: $n={RUN_D_N}$, shot budget", allc[allc.n == RUN_D_N])]
budgets = sorted(set(RUN_D_SHOTS) | {SHOTS_TRAIN}) + [0]
widths = {s: w for s, w in zip(budgets, [0.8, 1.2, 1.6, 1.1])}
dashes = {32: (0, (1, 1)), 128: (0, (3, 1)), SHOTS_TRAIN: "-", 0: (0, (6, 2, 1, 2))}

fig, axes = plt.subplots(2, len(panels), figsize=(IEEE_FULL * 0.7, 3.6), sharex=True, squeeze=False)
for j, (title, d) in enumerate(panels):
    for readout in ("global", "local"):
        for s in sorted(d.shots_train.unique()):
            g = d[(d.readout == readout) & (d.shots_train == s)].groupby("epoch")
            if not len(g):
                continue
            auc = g.test_auc.mean()
            axes[0, j].plot(auc.index, auc.values, color=COLORS[readout], ls=dashes.get(s, "-"),
                            lw=widths.get(s, 1.2))
            if s:
                ratio = g.exact_grad_norm.mean() / g.grad_norm.mean()
                axes[1, j].plot(ratio.index[1:], ratio.values[1:], color=COLORS[readout],
                                ls=dashes.get(s, "-"), lw=widths.get(s, 1.2))
    axes[0, j].set_title(title)
    axes[1, j].set_xlabel("epoch")
    axes[1, j].set_ylim(0, 1.05)
axes[0, 0].set_ylabel("test ROC-AUC")
axes[1, 0].set_ylabel("true / used gradient norm")
handles = [Line2D([], [], color=COLORS[r], label=LABELS[r].replace(" cost", " readout"))
           for r in ("global", "local")]
handles += [Line2D([], [], color="k", ls=dashes[s], lw=widths[s],
                   label=(f"{s} shots" if s else "exact gradients")) for s in budgets]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), fontsize=6.5)
fig.tight_layout(h_pad=0.5, w_pad=0.5)
save(fig, "shot_training_ext")
print("saved figures/shot_training_ext")
