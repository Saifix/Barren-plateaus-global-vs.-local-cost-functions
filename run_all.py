"""Regenerate every result and figure in order: python run_all.py [--skip-slow]

--skip-slow skips the two long trainings (03, 06; ~20 + ~25 min) and rebuilds
figures from the CSVs already in results/.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLOW = {"experiments/03_quantum_classifier.py", "experiments/06_shot_training.py"}
STEPS = [
    "experiments/00_make_folds.py",
    "experiments/01_variance_sweep.py",
    "experiments/02_classical_baselines.py",
    "experiments/03_quantum_classifier.py",
    "experiments/04_landscape.py",
    "experiments/05_leakage_check.py",
    "experiments/06_shot_training.py",
    "viz/plot_variance.py",
    "viz/plot_classifier.py",
    "viz/plot_shot_training.py",
    "viz/make_landscape_html.py",
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
