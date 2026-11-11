# Barren Plateaus: Global vs. Local Cost Functions

AI 631 Quantum Machine Learning, Fall 2026, Project #8 (QNN track)

## Research question

How does the gradient variance of a parameterized quantum circuit scale with
qubit count *n* and depth *L* for **global** and **local** cost functions?
Does a smarter initialization (identity-block or reduced-domain) help? And how
do these effects change the training of a small classifier in noise-free
simulation, under a noise model, and on IBM quantum hardware?

The global cost uses the all-zeros probability, `C_G = 1 - <0^n|ρ|0^n>`.
The local cost averages single-qubit probabilities,
`C_L = 1 - (1/n) Σ_j <0|ρ_j|0>`. Both come from the same Z-basis measurement.

## Plan

| Experiment | Details |
|---|---|
| Gradient-variance scaling | n = 2–14, L ∈ {1, ⌈log₂ n⌉, n}, 200 random parameter draws, global vs. local cost |
| Initialization | uniform, reduced-domain (∝ 1/√L), identity blocks |
| Training task | learn V(θ) mapping a fixed random state back to \|0…0⟩ |
| Classifier | Breast Cancer Wisconsin (PCA to n features), global vs. local readout; baselines: majority class, logistic regression, parameter-matched MLP |
| Noise | depolarizing (unital), amplitude damping (non-unital), IBM fake-backend noise model |
| Hardware | parameter-shift gradients on IBM hardware at n = 2, 4, 6 |

The hypotheses, falsifiers, references and week-by-week schedule are in the
synopsis.

## Repository layout

```
synopsis/
  synopsis.tex   Initial project synopsis (IEEEtran LaTeX source)
  synopsis.pdf   Compiled synopsis
```

Code, results and the final report will be added as the project progresses
(see the timeline below).

## Building the synopsis

The `.tex` file is self-contained (no separate `.bib`). Compile it with any
LaTeX distribution:

```
cd synopsis
pdflatex synopsis.tex
pdflatex synopsis.tex
```

You can also upload it to Overleaf.

## Timeline

| Due | Milestone |
|---|---|
| Nov 8 | Initial project synopsis |
| Nov 15 | Week 1: pipeline, folds, classical baselines, first quantum model |
| Nov 22 | Week 2: full simulator results, noise model, ablations |
| Nov 29 | Week 3: IBM hardware results |
| Dec 6 | Final report, presentation, tagged release |

## Author

Saif ur Rehman
