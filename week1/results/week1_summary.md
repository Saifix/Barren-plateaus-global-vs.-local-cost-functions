# Week 1 results summary

All rows: identical stratified 5-fold split (`data/folds.json`, seed 0); mean ± s.d. over folds (and 3 init seeds for MLP / quantum). In the per-n tables the quantum models are trained with cross-entropy and *exact* gradients (60 epochs) and then evaluated exactly and with shots; this run cannot show a barren plateau (see the H4 section below for why, and for the shot-noise experiment that can).

## n = 4 features / qubits (L = 2)

| model | accuracy | balanced acc. | ROC-AUC |
|---|---|---|---|
| Majority class | 0.627 ± 0.004 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.960 ± 0.021 | 0.950 ± 0.028 | 0.992 ± 0.009 |
| SVM (RBF, tuned) | 0.963 ± 0.011 | 0.956 ± 0.017 | 0.994 ± 0.005 |
| MLP (param.-matched) | 0.963 ± 0.015 | 0.955 ± 0.020 | 0.993 ± 0.007 |
| Quantum, global readout, exact eval | 0.841 ± 0.036 | 0.787 ± 0.050 | 0.976 ± 0.014 |
| Quantum, global readout, 2048-shot Aer eval | 0.844 ± 0.034 | 0.792 ± 0.048 | 0.976 ± 0.014 |
| Quantum, local readout, exact eval | 0.752 ± 0.037 | 0.668 ± 0.051 | 0.978 ± 0.010 |
| Quantum, local readout, 2048-shot Aer eval | 0.756 ± 0.041 | 0.673 ± 0.057 | 0.978 ± 0.010 |

## n = 6 features / qubits (L = 3)

| model | accuracy | balanced acc. | ROC-AUC |
|---|---|---|---|
| Majority class | 0.627 ± 0.004 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.967 ± 0.014 | 0.956 ± 0.020 | 0.993 ± 0.009 |
| SVM (RBF, tuned) | 0.967 ± 0.007 | 0.960 ± 0.007 | 0.995 ± 0.007 |
| MLP (param.-matched) | 0.974 ± 0.017 | 0.970 ± 0.021 | 0.995 ± 0.007 |
| Quantum, global readout, exact eval | 0.823 ± 0.041 | 0.762 ± 0.057 | 0.968 ± 0.019 |
| Quantum, global readout, 2048-shot Aer eval | 0.828 ± 0.038 | 0.769 ± 0.053 | 0.967 ± 0.020 |
| Quantum, local readout, exact eval | 0.724 ± 0.031 | 0.630 ± 0.043 | 0.963 ± 0.018 |
| Quantum, local readout, 2048-shot Aer eval | 0.724 ± 0.032 | 0.629 ± 0.045 | 0.962 ± 0.019 |

## n = 8 features / qubits (L = 3)

| model | accuracy | balanced acc. | ROC-AUC |
|---|---|---|---|
| Majority class | 0.627 ± 0.004 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.968 ± 0.018 | 0.959 ± 0.024 | 0.993 ± 0.009 |
| SVM (RBF, tuned) | 0.974 ± 0.018 | 0.969 ± 0.018 | 0.995 ± 0.007 |
| MLP (param.-matched) | 0.975 ± 0.017 | 0.973 ± 0.019 | 0.994 ± 0.007 |
| Quantum, global readout, exact eval | 0.811 ± 0.025 | 0.746 ± 0.035 | 0.958 ± 0.017 |
| Quantum, global readout, 2048-shot Aer eval | 0.814 ± 0.025 | 0.750 ± 0.035 | 0.957 ± 0.017 |
| Quantum, local readout, exact eval | 0.683 ± 0.026 | 0.575 ± 0.037 | 0.974 ± 0.012 |
| Quantum, local readout, 2048-shot Aer eval | 0.685 ± 0.030 | 0.577 ± 0.042 | 0.973 ± 0.012 |

## n = 10 features / qubits (L = 4)

| model | accuracy | balanced acc. | ROC-AUC |
|---|---|---|---|
| Majority class | 0.627 ± 0.004 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.965 ± 0.019 | 0.955 ± 0.026 | 0.993 ± 0.009 |
| SVM (RBF, tuned) | 0.974 ± 0.020 | 0.968 ± 0.022 | 0.995 ± 0.006 |
| MLP (param.-matched) | 0.978 ± 0.019 | 0.974 ± 0.022 | 0.995 ± 0.006 |
| Quantum, global readout, exact eval | 0.794 ± 0.026 | 0.724 ± 0.036 | 0.948 ± 0.018 |
| Quantum, global readout, 2048-shot Aer eval | 0.796 ± 0.025 | 0.726 ± 0.035 | 0.947 ± 0.018 |
| Quantum, local readout, exact eval | 0.677 ± 0.019 | 0.567 ± 0.027 | 0.962 ± 0.025 |
| Quantum, local readout, 2048-shot Aer eval | 0.678 ± 0.021 | 0.569 ± 0.030 | 0.960 ± 0.025 |

## H4 under shot noise (Brier loss, 30 epochs)

Same folds, seeds and hyper-parameters; only the gradient differs: exact (adjoint) vs. parameter-shift with 1024 shots per circuit. Primary metric: test ROC-AUC (threshold-free). 'True / used grad. norm' is the exact gradient norm divided by the norm of the gradient the optimizer actually received, averaged over training (1 = noiseless; << 1 = the update is mostly shot noise).

| n | readout | AUC, exact gradients | AUC, shot gradients | bal. acc., shot gradients | AUC, shot gradients, Aer 2048-shot eval | true / used grad. norm |
|---|---|---|---|---|---|---|
| 4 | global | 0.975 ± 0.013 | 0.974 ± 0.014 | 0.842 ± 0.056 | 0.973 ± 0.014 | 1.00 |
| 4 | local | 0.981 ± 0.009 | 0.981 ± 0.009 | 0.749 ± 0.055 | 0.980 ± 0.008 | 1.00 |
| 6 | global | 0.960 ± 0.019 | 0.960 ± 0.019 | 0.831 ± 0.061 | 0.960 ± 0.019 | 0.99 |
| 6 | local | 0.974 ± 0.017 | 0.975 ± 0.018 | 0.708 ± 0.062 | 0.973 ± 0.018 | 0.99 |
| 8 | global | 0.951 ± 0.021 | 0.951 ± 0.020 | 0.805 ± 0.049 | 0.950 ± 0.020 | 0.99 |
| 8 | local | 0.979 ± 0.014 | 0.979 ± 0.014 | 0.662 ± 0.029 | 0.978 ± 0.015 | 0.98 |

## Label-shuffle leakage check (n = 4)

Trained on permuted labels; held-out performance must fall to chance. Accuracy should equal the majority rate (0.627). AUC under shuffled labels is very spread out (s.d. ≈ 0.27), so its mean is only meaningful over many permutations: the logistic-regression row uses 40 per fold and reports the standard error; the quantum rows (one permutation per fold) are an accuracy check only.

| model | permutations | accuracy | ROC-AUC |
|---|---|---|---|
| logistic regression | 200 | 0.630 | 0.496 ± 0.017 (s.e.) |
| quantum-global | 5 | 0.627 | 0.602 ± 0.118 (s.d.) |
| quantum-local | 5 | 0.633 | 0.557 ± 0.203 (s.d.) |

## Initial gradient norm (epoch 0, mean over folds × seeds)

| n | global readout | local readout |
|---|---|---|
| 4 | 1.89 | 0.149 |
| 6 | 1.71 | 0.0833 |
| 8 | 2.64 | 0.116 |
| 10 | 2.3 | 0.0748 |

## Resources per circuit

| n | L | params | CZ | depth | circuits per gradient (parameter shift, per sample) |
|---|---|---|---|---|---|
| 4 | 2 | 16 | 6 | 9 | 32 |
| 6 | 3 | 36 | 15 | 13 | 72 |
| 8 | 3 | 48 | 21 | 13 | 96 |
| 10 | 4 | 80 | 36 | 17 | 160 |
