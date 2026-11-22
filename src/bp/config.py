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

# ---------------------------------------------------------------- Week 2
# H3: initializations on the identity-block HEA, ib_hea(n, D), D = ceil(log2 n)
INIT_NS = [2, 4, 6, 8, 10, 12, 14]
INIT_KINDS = ["uniform", "reduced", "identity"]
REDUCED_SCALES_ABLATION = [0.25, 0.5]   # x pi / sqrt(L); main run uses 1.0 (a = pi / sqrt(L))

# H3: the teacher-state training task
TASK_NS = [4, 6, 8, 10, 12]
TASK_SHOT_NS = [4, 6, 8, 10]          # shot-noise runs (n = 12: exact only, cost)
TASK_SEEDS = [SEED + 100 * r for r in range(10)]
TASK_ITERS = 300
TASK_LR = 0.05
TASK_SHOTS = 1024
TASK_TARGET = 0.1                       # success = C_G <= 0.1 (fidelity >= 0.9)

# H4 continued: Run C (n = 10, 1024 shots) and the shot-budget ablation (Run D)
RUN_C_NS = [10]
RUN_D_N = 8
RUN_D_SHOTS = [32, 128]

# Noise (density-matrix simulator, HEA, |0...0> input, uniform init)
NOISE_NS = [2, 4, 6, 8]
NOISE_DRAWS = 200
NOISE_P2 = 0.01                         # two-qubit channel strength; one-qubit = P2 / 10
NOISE_DEPTH_N = 6
NOISE_DEPTHS = [1, 2, 3, 4, 6, 8, 10, 12]
NOISE_STRENGTHS = [0.001, 0.003, 0.01, 0.03, 0.1]   # two-qubit strength, n = 6, L = 6

# IBM fake-backend noise model (Aer, density matrix, exact probabilities)
FAKE_BACKEND = "FakeFez"
FAKE_NS = [2, 4, 6, 8]
FAKE_DRAWS = 200
HW_SHOTS = 4096                         # used for the shot-noise floor (Week 3 setting)
