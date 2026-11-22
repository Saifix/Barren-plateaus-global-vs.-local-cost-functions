"""H3, part 2: does the initialization speed up training? (teacher-state task)

ib_hea(n, D), D = ceil(log2 n), learns to map |psi0> = V(theta*)^dagger |0...0>
back to |0...0> (bp.state_task). Grid: n in TASK_NS x training cost {global,
local} x init {uniform, reduced, identity} x TASK_SEEDS (10) x gradient mode:
  exact  adjoint gradients (control; Adam's per-coordinate scaling hides
         small-but-exact gradients, see Run A in Week 1)
  shots  parameter-shift gradients, TASK_SHOTS samples per circuit (primary)
Adam, TASK_LR, TASK_ITERS iterations. A seed fixes the target and the initial
angles, so all cells with the same seed are paired.
Success: C_G <= TASK_TARGET (fidelity >= 0.9), measured exactly, for both costs.

Shot-noise runs cost ~0.6 s per iteration at n = 10 but ~20-30 s at n = 12
(4096 amplitudes x 384 shifted circuits, memory-bound), so n = 12 is run with
exact gradients only (TASK_SHOT_NS limits the shot runs). Each finished run is
cached in results/cache/task/ so an interrupted run resumes where it stopped.

results/task_curves.csv   every 5th iteration of every run
results/task_final.csv    one row per run (iterations to target, final costs)
"""
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from bp.ansatz import depth_log
from bp.config import (TASK_ITERS, TASK_LR, TASK_NS, TASK_SEEDS, TASK_SHOT_NS, TASK_SHOTS,
                       TASK_TARGET)
from bp.data import ROOT
from bp.state_task import INITS, iters_to_target, train_state

MAX_WORKERS = int(os.environ.get("BP_WORKERS", 14))
LOG_EVERY = 5
CACHE = ROOT / "results" / "cache" / "task"


def run(job):
    n, cost, init, seed, mode = job
    f = CACHE / f"{n}_{cost}_{init}_{seed}_{mode}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return job, d["hist"], d["seconds"]
    t0 = time.time()
    hist = train_state(n, depth_log(n), cost, init, seed, mode=mode, shots=TASK_SHOTS,
                       iters=TASK_ITERS, lr=TASK_LR)
    dt = time.time() - t0
    f.write_text(json.dumps(dict(hist=hist, seconds=dt)))
    return job, hist, dt


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in itertools.product(TASK_NS, ("global", "local"), INITS, TASK_SEEDS, ("exact", "shots"))
            if j[4] == "exact" or j[0] in TASK_SHOT_NS]
    jobs.sort(key=lambda j: (j[4] != "shots", -j[0]))          # long jobs first
    curves, final = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for (n, cost, init, seed, mode), hist, dt in ex.map(run, jobs):
            key = dict(n=n, D=depth_log(n), cost=cost, init=init, seed=seed, mode=mode)
            curves += [dict(**key, **h) for h in hist if h["iter"] % LOG_EVERY == 0]
            hit = iters_to_target(hist, TASK_TARGET)
            final.append(dict(**key, iters_to_target=hit, reached=hit is not None,
                              C_G_init=hist[0]["C_G"], C_L_init=hist[0]["C_L"],
                              grad_norm_init=hist[0]["exact_grad_norm"],
                              C_G_final=hist[-1]["C_G"], C_L_final=hist[-1]["C_L"],
                              C_G_best=min(h["C_G"] for h in hist), seconds=dt))
            if n >= 10 and mode == "shots":
                print(f"n={n} {cost} {init} seed={seed}: hit={hit} C_G_final={hist[-1]['C_G']:.3f} "
                      f"({dt:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
    out = ROOT / "results"
    pd.DataFrame(curves).to_csv(out / "task_curves.csv", index=False)
    df = pd.DataFrame(final)
    df.to_csv(out / "task_final.csv", index=False)
    print(df.groupby(["mode", "n", "cost", "init"]).agg(
        reached=("reached", "mean"), median_iters=("iters_to_target", "median"),
        C_G_final=("C_G_final", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
