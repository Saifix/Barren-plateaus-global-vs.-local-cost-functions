"""Regenerate every result and figure in order: python run_all.py [--skip-slow]

Each week's scripts live in week<k>/experiments/ (computation -> week<k>/results/)
and week<k>/plots/ (figures and summaries -> week<k>/figures/, week<k>/results/).
--skip-slow skips the long experiments and rebuilds figures and summaries from
the CSVs already in week<k>/results/.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W1, W2, W3 = "week1/experiments/", "week2/experiments/", "week3/experiments/"
P1, P2, P3 = "week1/plots/", "week2/plots/", "week3/plots/"
SLOW = {W1 + "01_variance_sweep.py", W1 + "03_quantum_classifier.py", W1 + "06_classifier_shot_training.py",
        W2 + "07_init_variance.py", W2 + "08_teacher_task_training.py", W2 + "09_classifier_runs_c_d.py",
        W3 + "13_shot_budget_training.py"}
STEPS = [
    # Week 1: pipeline, baselines, first quantum model, variance sweep
    W1 + "00_make_folds.py",
    W1 + "01_variance_sweep.py",
    W1 + "02_classical_baselines.py",
    W1 + "03_quantum_classifier.py",
    W1 + "04_landscape.py",
    W1 + "05_leakage_check.py",
    W1 + "06_classifier_shot_training.py",
    P1 + "plot_variance.py",
    P1 + "plot_classifier.py",
    P1 + "plot_classifier_shot_training.py",
    P1 + "make_landscape_html.py",
    # Week 2: initializations, teacher task, Runs C/D, noise, FakeFez
    W2 + "07_init_variance.py",
    W2 + "08_teacher_task_training.py",
    W2 + "09_classifier_runs_c_d.py",
    W2 + "10_noise_variance.py",
    W2 + "11_fake_backend.py",
    P2 + "plot_init_variance.py",
    P2 + "plot_teacher_task.py",
    P2 + "plot_classifier_runs_c_d.py",
    P2 + "plot_noise.py",
    P2 + "make_summary.py",
    # Week 3: Q1 shot budget as a hyperparameter, Q2 analytical noise model
    W3 + "12_shot_budget.py",
    W3 + "13_shot_budget_training.py",
    W3 + "14_noise_model.py",
    P3 + "plot_shot_budget.py",
    P3 + "plot_noise_model.py",
    P3 + "make_summary.py",
]

skip_slow = "--skip-slow" in sys.argv
for step in STEPS:
    if skip_slow and step in SLOW:
        print(f"-- skipping {step}")
        continue
    print(f"== {step}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, str(ROOT / step)], check=True, cwd=ROOT)
    print(f"   done in {time.time() - t0:.0f}s", flush=True)
