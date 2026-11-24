"""One visual language for every figure, slide and animation in the project.

Global cost = warm vermilion, local cost = blue, classical = neutral greys.
Use ``COLORS[...]`` everywhere (matplotlib, Plotly, Manim) so the audience
never needs a legend after the first slide.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from .data import figures_dir

COLORS = {
    "global": "#E4572E",     # vermilion
    "local": "#1F77B4",      # blue
    "mlp": "#6A4C93",        # violet
    "logreg": "#7F7F7F",     # grey
    "svm_rbf": "#3B3B3B",    # charcoal
    "majority": "#B0B0B0",   # light grey
    "analytic": "#222222",
}
LABELS = {"global": "Global cost $C_G$", "local": "Local cost $C_L$",
          "mlp": "MLP (param.-matched)", "logreg": "Logistic regression",
          "svm_rbf": "SVM (RBF, tuned)", "majority": "Majority class"}

IEEE_COL = 3.5      # inches, single IEEE column
IEEE_FULL = 7.16    # inches, full page width


def use_style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 3.5,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, name, week):
    """Write week<week>/figures/<name>.pdf and .png."""
    out = figures_dir(week)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


# Week 2: initializations and noise models are told apart by line style and
# marker; colour keeps meaning "which cost".
INIT_STYLE = {"uniform": dict(ls="-", marker="o"),
              "reduced": dict(ls="--", marker="s"),
              "identity": dict(ls=":", marker="^")}
INIT_LABELS = {"uniform": "uniform", "reduced": r"reduced domain ($a=\pi/\sqrt{L}$)",
               "identity": "identity blocks"}
NOISE_STYLE = {"noiseless": dict(ls="-", marker="o"),
               "depolarizing": dict(ls="--", marker="s"),
               "amplitude_damping": dict(ls=":", marker="^"),
               "fake": dict(ls="-.", marker="D")}
NOISE_LABELS = {"noiseless": "noiseless", "depolarizing": "depolarizing (unital)",
                "amplitude_damping": "amplitude damping (non-unital)",
                "fake": "FakeFez noise model"}
