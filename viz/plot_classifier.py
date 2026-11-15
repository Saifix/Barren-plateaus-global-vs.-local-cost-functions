"""figures/training_curves.{pdf,png} and results/week1_summary.md."""
import matplotlib.pyplot as plt
import pandas as pd

from bp.ansatz import depth_log
from bp.config import CLF_NS, SHOTS
from bp.data import ROOT
from bp.qiskit_backend import resources
from bp.style import COLORS, IEEE_FULL, LABELS, save, use_style

use_style()
R = ROOT / "results"
q = pd.read_csv(R / "qclf_curves.csv")
m = pd.read_csv(R / "mlp_curves.csv")
qf = pd.read_csv(R / "qclf_final.csv")
base = pd.read_csv(R / "baselines.csv")
leak = pd.read_csv(R / "leakage_check.csv")              # quantum, from 03
leak_lr = pd.read_csv(R / "leakage_check_logreg.csv")    # classical, from 05

# ------------------------------------------------------------- curves
rows = [("test_auc", "test ROC-AUC"), ("test_bacc", "test balanced acc."), ("grad_norm", r"$\|\nabla\mathcal{L}\|$")]
fig, axes = plt.subplots(len(rows), len(CLF_NS), figsize=(IEEE_FULL, 4.6), sharex=True, sharey="row")
for j, n in enumerate(CLF_NS):
    for i, (col, ylabel) in enumerate(rows):
        ax = axes[i, j]
        for readout in ("global", "local"):
            g = q[(q.n == n) & (q.readout == readout)].groupby("epoch")[col].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], color=COLORS[readout], label=LABELS[readout].replace(" cost", " readout"))
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS[readout], alpha=0.18, lw=0)
        if col in m.columns:
            g = m[m.n == n].groupby("epoch")[col].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], color=COLORS["mlp"], ls="--", label=LABELS["mlp"])
        if col == "test_bacc":
            ax.axhline(0.5, color=COLORS["majority"], lw=0.8, ls=":")
        if col == "grad_norm":
            ax.set_yscale("log")
        if i == 0:
            ax.set_title(f"$n={n}$, $L={depth_log(n)}$")
        if j == 0:
            ax.set_ylabel(ylabel)
        if i == len(rows) - 1:
            ax.set_xlabel("epoch")
h, lab = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(h_pad=0.4, w_pad=0.4)
save(fig, "training_curves")


# ------------------------------------------------------------- tables
def ms(s):
    return f"{s.mean():.3f} ± {s.std():.3f}"


lines = ["# Week 1 results summary", "",
         "All rows: identical stratified 5-fold split (`data/folds.json`, seed 0); "
         "mean ± s.d. over folds (and 3 init seeds for MLP / quantum). In the per-n tables the "
         "quantum models are trained with cross-entropy and *exact* gradients (60 epochs) and then "
         "evaluated exactly and with shots; this run cannot show a barren plateau (see the H4 "
         "section below for why, and for the shot-noise experiment that can).", ""]
for n in CLF_NS:
    lines += [f"## n = {n} features / qubits (L = {depth_log(n)})", "",
              "| model | accuracy | balanced acc. | ROC-AUC |", "|---|---|---|---|"]
    for model in ("majority", "logreg", "svm_rbf", "mlp"):
        s = base[(base.n == n) & (base.model == model)]
        lines.append(f"| {LABELS[model]} | {ms(s.acc)} | {ms(s.bacc)} | {ms(s.auc)} |")
    for readout in ("global", "local"):
        s = qf[(qf.n == n) & (qf.readout == readout)]
        lines.append(f"| Quantum, {readout} readout, exact eval | {ms(s.exact_acc)} | {ms(s.exact_bacc)} | {ms(s.exact_auc)} |")
        lines.append(f"| Quantum, {readout} readout, {SHOTS}-shot Aer eval | {ms(s.shots_acc)} | {ms(s.shots_bacc)} | {ms(s.shots_auc)} |")
    lines.append("")

shot_path = R / "shot_final.csv"
if shot_path.exists():
    from bp.config import SHOTS_TRAIN, SHOT_EPOCHS
    sf = pd.read_csv(shot_path)
    lines += [f"## H4 under shot noise (Brier loss, {SHOT_EPOCHS} epochs)", "",
              f"Same folds, seeds and hyper-parameters; only the gradient differs: exact (adjoint) "
              f"vs. parameter-shift with {SHOTS_TRAIN} shots per circuit. Primary metric: test ROC-AUC "
              f"(threshold-free). 'True / used grad. norm' is the exact gradient norm divided by the "
              f"norm of the gradient the optimizer actually received, averaged over training "
              f"(1 = noiseless; << 1 = the update is mostly shot noise).", "",
              "| n | readout | AUC, exact gradients | AUC, shot gradients | bal. acc., shot gradients "
              "| AUC, shot gradients, Aer 2048-shot eval | true / used grad. norm |",
              "|---|---|---|---|---|---|---|"]
    sc = pd.read_csv(R / "shot_curves.csv")
    for n in sorted(sf.n.unique()):
        for readout in ("global", "local"):
            ex = sf[(sf.n == n) & (sf.readout == readout) & (sf["mode"] == "exact")]
            sh = sf[(sf.n == n) & (sf.readout == readout) & (sf["mode"] == "shots")]
            cc = sc[(sc.n == n) & (sc.readout == readout) & (sc["mode"] == "shots") & (sc.epoch > 0)]
            ratio = (cc.exact_grad_norm / cc.grad_norm).mean()
            lines.append(f"| {n} | {readout} | {ms(ex.exact_auc)} | {ms(sh.exact_auc)} | "
                         f"{ms(sh.exact_bacc)} | {ms(sh.shots_auc)} | {ratio:.2f} |")
    lines.append("")

lines += ["## Label-shuffle leakage check (n = 4)", "",
          "Trained on permuted labels; held-out performance must fall to chance. "
          "Accuracy should equal the majority rate (0.627). AUC under shuffled labels is very "
          "spread out (s.d. ≈ 0.27), so its mean is only meaningful over many permutations: "
          "the logistic-regression row uses 40 per fold and reports the standard error; "
          "the quantum rows (one permutation per fold) are an accuracy check only.", "",
          "| model | permutations | accuracy | ROC-AUC |", "|---|---|---|---|"]
lines.append(f"| logistic regression | {len(leak_lr)} | {leak_lr.acc.mean():.3f} | "
             f"{leak_lr.auc.mean():.3f} ± {leak_lr.auc.std() / len(leak_lr) ** 0.5:.3f} (s.e.) |")
for model, s in leak.groupby("model"):
    lines.append(f"| {model} | {len(s)} | {s.acc.mean():.3f} | {ms(s.auc)} (s.d.) |")
lines += ["", "## Initial gradient norm (epoch 0, mean over folds × seeds)", "",
          "| n | global readout | local readout |", "|---|---|---|"]
for n in CLF_NS:
    g = qf[qf.n == n].groupby("readout").init_grad_norm.mean()
    lines.append(f"| {n} | {g['global']:.3g} | {g['local']:.3g} |")
lines += ["", "## Resources per circuit", "",
          "| n | L | params | CZ | depth | circuits per gradient (parameter shift, per sample) |",
          "|---|---|---|---|---|---|"]
for n in CLF_NS:
    r = resources(n, depth_log(n))
    lines.append(f"| {n} | {r['layers']} | {r['params']} | {r['cz']} | {r['depth']} | {r['circuits_per_gradient_ps']} |")
(R / "week1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
