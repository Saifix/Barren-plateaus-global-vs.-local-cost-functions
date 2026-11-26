"""Q1, training: the shot budget as a hyperparameter on the teacher task.

Same task and protocol as week2/experiments/08_teacher_task_training.py
(ib_hea, D = ceil(log2 n), Adam 0.05, 300 iterations, success = C_G <= 0.1),
uniform initialization only, both training costs, parameter-shift gradients with
N shots per circuit for N in SHOT_GRID:
  n in SHOT_TASK_NS, 10 seeds, every N (the N = 1024 runs are Week 2's, read
  from its cache when present);
  n = 12, SHOT_TASK_N12_SEEDS seeds, N in SHOT_TASK_N12_SHOTS: tests the Week 2
  extrapolation that 1024 shots stop being enough for the global cost near n = 12.

Each run is cached in results/cache/shots/ so an interrupted run resumes.
results/shot_training_curves.csv, results/shot_training_final.csv
"""
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from bp.ansatz import depth_log
from bp.config import (SHOT_GRID, SHOT_TASK_N12_SEEDS, SHOT_TASK_N12_SHOTS, SHOT_TASK_NS, TASK_ITERS,
                       TASK_LR, TASK_SEEDS, TASK_TARGET)
from bp.data import results_dir
from bp.state_task import iters_to_target, train_state

MAX_WORKERS = int(os.environ.get("BP_WORKERS", 13))
LOG_EVERY = 5
CACHE = results_dir(3) / "cache" / "shots"
WEEK2_CACHE = results_dir(2) / "cache" / "task"     # Week 2's 1024-shot uniform-init runs


def run(job):
    n, cost, shots, seed = job
    f = CACHE / f"{n}_{cost}_{shots}_{seed}.json"
    old = WEEK2_CACHE / f"{n}_{cost}_uniform_{seed}_shots.json"
    for src in (f, old if shots == 1024 else None):
        if src is not None and src.exists():
            d = json.loads(src.read_text())
            return job, d["hist"], d["seconds"]
    t0 = time.time()
    hist = train_state(n, depth_log(n), cost, "uniform", seed, mode="shots", shots=shots,
                       iters=TASK_ITERS, lr=TASK_LR)
    dt = time.time() - t0
    f.write_text(json.dumps(dict(hist=hist, seconds=dt)))
    return job, hist, dt


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = list(itertools.product(SHOT_TASK_NS, ("global", "local"), SHOT_GRID, TASK_SEEDS))
    jobs += list(itertools.product([12], ("global", "local"), SHOT_TASK_N12_SHOTS,
                                   TASK_SEEDS[:SHOT_TASK_N12_SEEDS]))
    jobs.sort(key=lambda j: -j[0])                                  # long jobs first
    curves, final = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for (n, cost, shots, seed), hist, dt in ex.map(run, jobs):
            key = dict(n=n, D=depth_log(n), cost=cost, shots=shots, seed=seed)
            curves += [dict(**key, **h) for h in hist if h["iter"] % LOG_EVERY == 0]
            hit = iters_to_target(hist, TASK_TARGET)
            final.append(dict(**key, iters_to_target=hit, reached=hit is not None,
                              C_G_init=hist[0]["C_G"], grad_norm_init=hist[0]["exact_grad_norm"],
                              C_G_final=hist[-1]["C_G"], C_L_final=hist[-1]["C_L"],
                              C_G_best=min(h["C_G"] for h in hist), seconds=dt))
            if n >= 10:
                print(f"n={n} {cost} shots={shots} seed={seed}: hit={hit} "
                      f"C_G_final={hist[-1]['C_G']:.3f} ({dt:.0f}s, total {time.time() - t0:.0f}s)",
                      flush=True)
    out = results_dir(3)
    pd.DataFrame(curves).to_csv(out / "shot_training_curves.csv", index=False)
    df = pd.DataFrame(final)
    df.to_csv(out / "shot_training_final.csv", index=False)
    print(df.groupby(["n", "cost", "shots"]).agg(
        reached=("reached", "mean"), median_iters=("iters_to_target", "median"),
        C_G_final=("C_G_final", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
