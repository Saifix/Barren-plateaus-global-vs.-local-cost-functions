"""figures/shot_training.{pdf,png}: H4 under shot noise vs. the exact-gradient control."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from bp.ansatz import depth_log
from bp.config import SHOT_NS, SHOTS_TRAIN
from bp.data import ROOT
from bp.style import COLORS, IEEE_FULL, LABELS, save, use_style

use_style()
c = pd.read_csv(ROOT / "results" / "shot_curves.csv")

fig, axes = plt.subplots(2, len(SHOT_NS), figsize=(IEEE_FULL, 3.6), sharex=True, sharey="row")
for j, n in enumerate(SHOT_NS):
    ax_auc, ax_g = axes[0, j], axes[1, j]
    for readout in ("global", "local"):
        col = COLORS[readout]
        for mode, ls, lw in (("shots", "-", 1.5), ("exact", ":", 1.1)):
            g = c[(c.n == n) & (c.readout == readout) & (c["mode"] == mode)].groupby("epoch")
            auc = g.test_auc.agg(["mean", "std"])
            ax_auc.plot(auc.index, auc["mean"], ls=ls, lw=lw, color=col)
            if mode == "shots":
                ax_auc.fill_between(auc.index, auc["mean"] - auc["std"], auc["mean"] + auc["std"],
                                    color=col, alpha=0.15, lw=0)
                ax_g.plot(auc.index, g.exact_grad_norm.mean(), "-", color=col, lw=1.5)
                ax_g.plot(auc.index, g.grad_norm.mean(), "--", color=col, lw=1.1)
    ax_auc.axhline(0.5, color=COLORS["majority"], lw=0.8, ls=":")
    ax_auc.set_title(f"$n={n}$, $L={depth_log(n)}$")
    ax_g.set_yscale("log")
    ax_g.set_xlabel("epoch")
axes[0, 0].set_ylabel("test ROC-AUC")
axes[1, 0].set_ylabel(r"$\|\nabla\mathcal{L}\|$ (shot runs)")
handles = [Line2D([], [], color=COLORS["global"], label=LABELS["global"].replace(" cost", " readout")),
           Line2D([], [], color=COLORS["local"], label=LABELS["local"].replace(" cost", " readout")),
           Line2D([], [], color="k", ls="-", label=f"top: trained with {SHOTS_TRAIN}-shot gradients"),
           Line2D([], [], color="k", ls=":", label="top: trained with exact gradients (control)"),
           Line2D([], [], color="k", ls="-", label="bottom: true gradient norm"),
           Line2D([], [], color="k", ls="--", label="bottom: shot estimate (signal + noise)")]
fig.legend(handles=handles, loc="upper center", ncol=3,  # fills column-wise
           bbox_to_anchor=(0.5, 1.10))
fig.tight_layout(h_pad=0.5, w_pad=0.5)
save(fig, "shot_training")
print("saved figures/shot_training")
