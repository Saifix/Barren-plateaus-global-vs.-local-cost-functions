"""Regenerate every result and figure in order: python run_all.py [--skip-slow]

--skip-slow skips the long runs (01, 03, 06, 07, 08, 09) and rebuilds figures
and summaries from the CSVs already in results/.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLOW = {"experiments/01_variance_sweep.py", "experiments/03_quantum_classifier.py",
        "experiments/06_shot_training.py", "experiments/07_init_variance.py",
        "experiments/08_state_training.py", "experiments/09_shot_training_ext.py"}
STEPS = [
    "experiments/00_make_folds.py",
    "experiments/01_variance_sweep.py",
    "experiments/02_classical_baselines.py",
    "experiments/03_quantum_classifier.py",
    "experiments/04_landscape.py",
    "experiments/05_leakage_check.py",
    "experiments/06_shot_training.py",
    "experiments/07_init_variance.py",
    "experiments/08_state_training.py",
    "experiments/09_shot_training_ext.py",
    "experiments/10_noise_variance.py",
    "experiments/11_fake_backend.py",
    "viz/plot_variance.py",
    "viz/plot_classifier.py",
    "viz/plot_shot_training.py",
    "viz/make_landscape_html.py",
    "viz/plot_init.py",
    "viz/plot_task.py",
    "viz/plot_shot_training_ext.py",
    "viz/plot_noise.py",
    "viz/week2_summary.py",
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
