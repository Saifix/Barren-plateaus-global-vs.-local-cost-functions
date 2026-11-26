"""figures/noise_model.{pdf,png}: Q2, the analytical Pauli-path model vs. simulation,
the gap between the costs under noise, and the light-cone mechanism."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from bp.config import NOISE_P2
from bp.data import results_dir
from bp.style import COLORS, IEEE_FULL, LABELS, NOISE_LABELS, NOISE_STYLE, save, use_style

use_style()
R = results_dir(3)
val = pd.read_csv(R / "noise_model_validation.csv")
mp = pd.read_csv(R / "noise_model_map.csv")
wt = pd.read_csv(R / "noise_model_weights.csv")
sim2 = pd.read_csv(results_dir(2) / "noise_variance.csv")

fig, axes = plt.subplots(1, 4, figsize=(IEEE_FULL, 2.45))

# (a) every validation point: model vs simulation
ax = axes[0]
marks = {"noiseless": "o", "depolarizing": "s", "amplitude_damping": "^"}
for (model, cost), d in val.groupby(["model", "cost"]):
    yerr = [d.sim - d.sim_lo, d.sim_hi - d.sim]
    ax.errorbar(d.model_value, d.sim, yerr=yerr, fmt=marks[model], color=COLORS[cost], mfc="white",
                ms=2.6, lw=0.5, capsize=0, alpha=0.85)
lim = [val[["sim", "model_value"]].min().min() * 0.5, val[["sim", "model_value"]].max().max() * 2]
ax.plot(lim, lim, color="k", lw=0.6)
ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim)
ax.set_xlabel("model Var (exact average)")
ax.set_ylabel("simulated Var (95% CI)")
ax.set_title("(a) model vs. simulation")
ax.legend(handles=[Line2D([], [], color="0.3", ls="", marker=m, mfc="white", label=NOISE_LABELS.get(k, k).split(" (")[0])
                   for k, m in marks.items()], fontsize=5.3, loc="lower right")

# (b) depth study, n = 6: simulation (markers) and model (lines)
ax = axes[1]
d6 = val[(val.study == "depth") & (val.stat == "var_mean")]
for (model, cost), d in d6.groupby(["model", "cost"]):
    d = d.sort_values("L")
    st = NOISE_STYLE[model]
    ax.plot(d.L, d.model_value, color=COLORS[cost], ls=st["ls"], lw=1)
    ax.errorbar(d.L, d.sim, yerr=[d.sim - d.sim_lo, d.sim_hi - d.sim], fmt=st["marker"], color=COLORS[cost],
                mfc="white", ms=2.6, lw=0.5, capsize=0)
ax.set_yscale("log")
ax.set_xlabel("layers $L$")
ax.set_ylabel("Var averaged over $k$")
ax.set_title(f"(b) depth, $n=6$")

# (c) how the gap changes: (Var_L / Var_G)_noisy / (Var_L / Var_G)_noiseless, p2 = 0.01
ax = axes[2]
piv = mp.pivot_table(index=["n", "L", "p2"], columns="cost", values="var_k").reset_index()
piv["gap"] = piv.local / piv["global"]
base = piv[piv.p2 == 0].set_index(["n", "L"]).gap
cmap = plt.get_cmap("viridis")
ns = sorted(piv.n.unique())
for i, n in enumerate(ns):
    for p2, ls in ((0.01, "-"), (0.03, "--")):
        d = piv[(piv.n == n) & (piv.p2 == p2)].sort_values("L")
        ax.plot(d.L, d.gap.values / base.loc[[(n, L) for L in d.L]].values, color=cmap(i / max(1, len(ns) - 1)),
                ls=ls, lw=1, marker="o" if ls == "-" else None, ms=2)
ax.axhline(1, color="k", lw=0.6)
ax.set_yscale("log")
ax.set_xlabel("layers $L$")
ax.set_ylabel("gap noisy / gap noiseless")
ax.set_title("(c) gap growth (depol.)")
ax.legend(handles=[Line2D([], [], color=cmap(i / max(1, len(ns) - 1)), label=f"$n={n}$") for i, n in enumerate(ns)]
          + [Line2D([], [], color="k", ls="-", label="$p_2=0.01$"), Line2D([], [], color="k", ls="--", label="$p_2=0.03$")],
          fontsize=5.5, ncol=2, loc="upper left")

# (d) mechanism: Pauli weight of the backward-propagated observable
ax = axes[3]
for (n, L), ls in (((10, 10), "-"), ((8, 8), ":")):
    for cost in ("global", "local"):
        d = wt[(wt.n == n) & (wt.L == L) & (wt.cost == cost)]
        ax.plot(d.layer_from_end, d.mean_weight, color=COLORS[cost], ls=ls, marker="o", ms=2.2, lw=1)
ax.set_xlabel("layers before measurement")
ax.set_ylabel("mean Pauli weight")
ax.set_title("(d) qubits hit by noise")
ax.legend(handles=[Line2D([], [], color=COLORS[c], label=LABELS[c]) for c in ("global", "local")]
          + [Line2D([], [], color="k", ls="-", label="$n=10$"), Line2D([], [], color="k", ls=":", label="$n=8$")],
          fontsize=5.3, loc="lower right")
fig.tight_layout(w_pad=0.6)
save(fig, "noise_model", 3)
print("saved week3/figures/noise_model")
