# Barren Plateaus: Global vs. Local Cost Functions

AI 631 Quantum Machine Learning, Fall 2026, Project #8 (QNN track)

![Gradient variance vs. qubits for the global and local cost](week1/figures/variance_scaling.png)

**Interactive 3D explorer:** open [`week1/figures/landscape_3d.html`](week1/figures/landscape_3d.html)
in a browser. Drag to rotate, press *Play n* to watch the global landscape flatten.
Deep links work for slides, e.g. `landscape_3d.html?n=12&mode=random`.

## Research question

How does the gradient variance of a parameterized quantum circuit scale with
qubit count *n* and depth *L* for **global** and **local** cost functions?
Does a smarter initialization (identity-block or reduced-domain) help? And how
do these effects change the training of a small classifier in noise-free
simulation, under a noise model, and on IBM quantum hardware?

The global cost uses the all-zeros probability, `C_G = 1 - <0^n|ρ|0^n>`.
The local cost averages single-qubit probabilities,
`C_L = 1 - (1/n) Σ_j <0|ρ_j|0>`. Both come from the same Z-basis measurement.

## Week 1 status (Nov 15)

| Deliverable | Where |
|---|---|
| Verified simulator + exact (adjoint) gradients, analytic unit test | `src/bp/sim.py`, `tests/` (22 tests) |
| Data pipeline, fixed folds (NB11 protocol + fold-wise PCA) | `src/bp/data.py`, `data/folds.json` |
| Classical baselines (majority, logistic, tuned RBF-SVM, param-matched MLP) | `week1/experiments/02_classical_baselines.py` |
| First quantum model, global vs. local readout, exact + 2048-shot Aer evaluation (Run A) | `week1/experiments/03_quantum_classifier.py` |
| Shot-noise training, 1024-shot parameter shift vs. exact control (Run B, H4) | `week1/experiments/06_classifier_shot_training.py` |
| Label-shuffle leakage checks | `week1/experiments/03_…`, `05_leakage_check.py` |
| Noise-free gradient-variance sweep, n = 2…14, three depths | `week1/experiments/01_variance_sweep.py` |
| Interactive 3D cost-landscape explorer | `week1/experiments/04_landscape.py`, `week1/plots/make_landscape_html.py` |
| Report: introduction and methods | `report/report.tex` |

Headline numbers are in [`week1/results/week1_summary.md`](week1/results/week1_summary.md).
In brief:

* **H1 supported.** At L = ⌈log₂ n⌉ the fitted decay rate of Var[∂C] is
  b = 1.20 [1.15, 1.26] for the global cost vs. 0.37 [0.36, 0.38] for the
  local cost (95% bootstrap CIs; 2000 parameter draws per point for n ≥ 10,
  because the global gradient is heavy-tailed). At n = 14 the variances differ
  by about 6800×.
* **H2 supported.** The local-cost decay rate rises from 0.37 to 0.44 [0.43, 0.46]
  at L = n (global: 1.20 → 1.33).
* **H4 not supported at n ≤ 8.** Run A (cross-entropy, exact gradients) cannot
  show a plateau: `-log P(0^n)` rescales the global readout's small gradient by
  `1/P`. Run B removes both masks: Brier loss, 1024-shot parameter-shift gradients,
  and an exact-gradient control on the same folds and seeds. At initialization
  the global readout's gradient is 2–5× smaller than the local one (0.008 vs.
  0.040 at n = 6). With a 32-sample batch, the shot estimate still keeps
  98–100% of the true gradient norm. Shot-trained and exact-trained models reach
  the same test AUC (global 0.974 → 0.951, local 0.981 → 0.979 for n = 4 → 8).
  The small global-vs.-local gap appears identically without shot noise, so it is
  not a shot-noise effect. Testing H4 needs larger n or fewer shots (Week 2).
  All quantum models stay below logistic regression, the RBF-SVM and the MLP
  (AUC ≈ 0.99).

![Training under shot noise vs. exact gradients](week1/figures/shot_training.png)

## Week 2 status (Nov 22)

| Deliverable | Where |
|---|---|
| Density-matrix simulator with exact adjoint gradients (depolarizing, amplitude damping) | `src/bp/density.py`, `tests/test_density.py` |
| Identity-block circuit `ib_hea` and the three initializations | `src/bp/ansatz.py`, `tests/test_initialization_and_task.py` |
| H3 part 1: initial gradient variance by initialization, n = 2…14, width ablation | `week2/experiments/07_init_variance.py` |
| H3 part 2: teacher-state training, 10 seeds, shot-noise + exact gradients | `week2/experiments/08_teacher_task_training.py`, `src/bp/state_task.py` |
| H4 continued: Run C (n = 10) and Run D (32/128 shots, n = 8) | `week2/experiments/09_classifier_runs_c_d.py` |
| Noise study: n, depth and strength sweeps | `week2/experiments/10_noise_variance.py` |
| IBM `FakeFez` device noise model (Aer density matrix + readout error) | `week2/experiments/11_fake_backend.py` |
| Report: results section draft | `report/report.tex` |

Headline numbers are in [`week2/results/week2_summary.md`](week2/results/week2_summary.md);
a plain-language overview is in [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).
In brief:

* **H1/H2 refit over the pre-registered n = 4…14** (Week 1 included n = 2):
  b_G = 1.17 vs. b_L = 0.29 at L = ⌈log₂ n⌉; at L = n, 1.33 vs. 0.37. Same conclusions.
* **H3, variance criterion not met.** At n ≥ 10, identity blocks raise the initial
  global-cost variance by 1.5–2.5× and reduced-domain starts by 1.1–1.3×, far
  from the pre-registered 10×. The target overlap itself is exponentially small.
* **H3, training criterion not met.** On the teacher-state task (10 seeds) neither
  start trains faster at n = 10 (1024-shot gradients) or n = 12 (exact); all
  Mann–Whitney p ≥ 0.36. At these sizes the global cost trains as well as the local
  one, because the shot noise of P(0ⁿ) shrinks with P itself.
* **H4, pre-registered falsifier not triggered, but the gap is not a plateau effect.**
  At n = 10 (Run C) the local readout's test AUC exceeds the global one's by
  +0.029 [0.016, 0.042] (paired, > 1 s.d.). The gap is the same with exact
  gradients (+0.026), both readouts learn equally fast, and cutting the budget to
  32 shots (Run D, n = 8) barely changes either readout. It is a readout-capacity
  difference, not slower training.
* **Noise.** At shallow depth, 1% two-qubit noise removes at most 20% of the
  variance; the global/local gap is orders of magnitude larger. In deep circuits
  (n = 6, L = 12), depolarizing noise removes 67% (global) and 53% (local).
* **FakeFez.** Variances stay within 15% of noiseless up to n = 8. The 4096-shot
  floor is ≥ 180× below the signal at n ≤ 6, so the Week 3 hardware run is feasible.

![Initial gradient variance by initialization](week2/figures/init_variance.png)

## Week 3 status (in progress): two focused questions

After the Week 2 review the project is narrowed to two questions:
**Q1** how many shots does each cost need to train (shots as a hyperparameter), and
**Q2** do depth and hardware noise erase the local cost's advantage (analytical
modelling of random circuits plus simulation).

| Deliverable | Where |
|---|---|
| Exact Pauli-path model of the gradient variance of random-angle circuits, with gate noise | `src/bp/pauli_path.py`, `tests/test_pauli_path.py` |
| Q1: shots per circuit needed for a measurable gradient, n = 2…14 | `week3/experiments/12_shot_budget.py` |
| Q1: teacher-task training at 256 / 1024 / 5000 / 10000 shots, n = 6…12 | `week3/experiments/13_shot_budget_training.py` |
| Q2: model vs. every Week 1/2 simulation, regime map to n = 10, L = 20 | `week3/experiments/14_noise_model.py` |
| Progress-update slides | `presentations/` |

Headline numbers: [`week3/results/week3_summary.md`](week3/results/week3_summary.md).

* **Q1.** The local cost needs 11–29 shots per circuit for a gradient signal above
  the shot noise, up to 14 qubits (×1.5 per 2 qubits). The global cost needs ×3.5
  more per 2 qubits: 790 at n = 12 and 3,740 at n = 14. That is 5–4,600× fewer than
  the worst-case estimate, because the shot noise of a small probability P(0ⁿ) is
  itself small. 1024 shots stop being enough for the global cost between 12 and 14 qubits.
  Training confirms it: up to 12 qubits, 1024 shots train both costs (≥ 60% of seeds
  reach the target at every n); 10,000 shots mainly cut the steps (global, n = 12:
  271 → 105 median steps) and do not change which cost wins.
* **Q2.** The model reproduces the simulated noise retention to 0.4% (median) for
  depolarizing noise. Unital noise **widens** the local advantage: the local cost's
  operator stays inside its light cone, so it meets the same number of noisy gates
  at any n, while the global one spreads over ≈ 0.6 n qubits per layer. At n = 10,
  L = 20 and 1% noise the gap grows 4.5×. Amplitude damping is the exception: its
  pull towards |0…0⟩ (the global target) partly favours the global cost.

![Q1: shot budget](week3/figures/shot_budget.png)

![Q2: analytical noise model](week3/figures/noise_model.png)

## Repository layout

```
week1/             Week 1 (Nov 15): pipeline, baselines, first quantum model, variance sweep
  experiments/       00 folds · 01 variance sweep · 02 classical baselines · 03 quantum classifier (Run A)
                     04 landscape · 05 leakage check · 06 classifier shot-noise training (Run B)
  plots/             figure scripts, Week 1 summary, 3D explorer generator
  results/           CSV outputs + week1_summary.md
  figures/           PDF/PNG figures + landscape_3d.html
week2/             Week 2 (Nov 22): initializations, teacher task, Runs C/D, noise, FakeFez
  experiments/       07 init variance · 08 teacher-task training · 09 classifier Runs C/D
                     10 noise variance · 11 FakeFez noise model
  plots/             figure scripts + make_summary.py
  results/           CSV outputs + week2_summary.md (results/cache/: run checkpoints, not committed)
  figures/           PDF/PNG figures
week3/             Week 3: Q1 shot budget (12, 13) and Q2 analytical noise model (14)
  experiments/  plots/  results/  figures/
src/bp/            shared library used by every week: simulators, circuits & costs, data,
                   baselines, classifier, H3 task, Qiskit circuits, plot style, paths
tests/             automated checks that the library is correct (pytest): Qiskit/Aer cross-checks,
                   finite differences, parameter shift, closed forms. They produce no results.
data/folds.json    the one fold object used by every table
report/            IEEE report (report.tex, refs.bib) and the submitted weekly PDFs
presentations/     progress-update slides
synopsis/          initial project synopsis
run_all.py         runs every experiment and plot script, week by week
```

**Experiments vs. tests.** A script in `week*/experiments/` *produces results*
(CSV files in that week's `results/`); a script in `week*/plots/` turns them into
figures and summaries. A file in `tests/` *checks the code* (e.g. that the
simulator's gradients match Qiskit and finite differences) and produces nothing.

## Reproduce

```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest
.venv\Scripts\python run_all.py              # everything; Week 2 adds ~1 day on 16 threads
.venv\Scripts\python run_all.py --skip-slow  # skip 01, 03, 06–09, rebuild figures from week*/results/
```

Approximate times (16 logical cores): the variance sweep (01) takes ~2 h,
dominated by n = 14 with 2000 draws. Run A (03) takes ~20 min and Run B (06)
~25 min. Week 2: 07 ~1 h, 10 and 11 ~30 min each; 08 (teacher-state training)
~5 h and 09 (Runs C/D) ~14 h, dominated by 1024-shot parameter-shift training at
n = 10. Both write a checkpoint per finished run to `week2/results/cache/` and resume
from it when restarted; `BP_WORKERS` sets the number of worker processes.
Everything else takes a minute or less. One seed constant (`bp.SEED = 0`);
every experimental constant is in `src/bp/config.py`. GitHub Actions runs the
test suite on Python 3.12 and 3.14.

## Building the report

`report/report.tex` uses IEEEtran, TikZ with the `quantikz2` library, and BibTeX
(`refs.bib`), and reads figures from `../week1/figures/` and `../week2/figures/`. On Overleaf, upload
`report/`, `week1/figures/` and `week2/figures/` with the same layout. Locally:
`pdflatex report && bibtex report && pdflatex report && pdflatex report`, or
`tectonic report.tex` (XeTeX; the preamble then switches to TeX Gyre Termes,
because IEEEtran's Times has no Unicode shapes).

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
