"""Every experimental constant in one place (the report's Methods quotes these)."""
from . import N_FOLDS, SEED  # noqa: F401

# Gradient-variance sweep
VAR_NS = [2, 4, 6, 8, 10, 12, 14]
VAR_DEPTHS = ["1", "log", "n"]       # L = 1, ceil(log2 n), n
VAR_DRAWS = 200
VAR_DRAWS_LARGE = 2000               # for n >= VAR_LARGE_FROM: the global-cost gradient is
VAR_LARGE_FROM = 10                  # heavy-tailed (top 5 of 1000 draws carry ~65% of the
                                     # variance), so 200 draws give a CI spanning >10x at n = 14
VAR_K_FIXED = 0                      # R_Y on qubit 0, first layer
N_BOOT = 1000


def var_draws(n: int) -> int:
    return VAR_DRAWS_LARGE if n >= VAR_LARGE_FROM else VAR_DRAWS

# Classifier
CLF_NS = [4, 6, 8, 10]               # PCA components = qubits; L = ceil(log2 n)
EPOCHS = 60
BATCH = 32
LR_QUANTUM = 0.05
LR_MLP = 0.01
INIT_SEEDS = [SEED + 100 * r for r in range(3)]   # spaced seeds (NB11)
SHOTS = 2048                         # sampled (Aer) evaluation of every trained model

# Shot-based training (H4): finite-shot parameter-shift gradients, Brier loss
SHOT_NS = [4, 6, 8]
SHOTS_TRAIN = 1024                   # shots per circuit per gradient evaluation
SHOT_EPOCHS = 30
