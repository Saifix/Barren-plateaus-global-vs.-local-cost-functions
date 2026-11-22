"""H4 continued: the Run B protocol at larger n (Run C) and with fewer shots (Run D).

Week 1 (Run B, n = 4, 6, 8, 1024 shots) found no shot-noise effect: with a
32-sample batch the shot estimate kept 98-100% of the true gradient norm.
Two ways to make the gradient signal small compared with its shot noise:
  Run C  n in RUN_C_NS (10 qubits, L = 4, 80 parameters), 1024 shots + exact control
  Run D  n = RUN_D_N (8), shots in RUN_D_SHOTS (32, 128); the exact control
         and the 1024-shot runs are those of Run B (same folds, seeds, settings)
Everything else is identical to experiments/06_shot_training.py: Brier loss,
Adam lr 0.05, batch 32, 30 epochs, 5 folds x 3 init seeds, test ROC-AUC primary,
plus a 2048-shot Aer evaluation of every trained model.

Each finished model is cached in results/cache/shot_ext/, so an interrupted
run resumes where it stopped.

results/shot_ext_curves.csv, results/shot_ext_final.csv
"""
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bp import SEED
from bp.ansatz import depth_log
from bp.config import (BATCH, INIT_SEEDS, LR_QUANTUM, RUN_C_NS, RUN_D_N, RUN_D_SHOTS,
                       SHOT_EPOCHS, SHOTS, SHOTS_TRAIN)
from bp.data import ROOT, load_data, load_folds, make_preprocessor
from bp.metrics import scores
from bp.qclassifier import QuantumClassifier

X, y = load_data()
FOLDS = load_folds()
MAX_WORKERS = int(os.environ.get("BP_WORKERS", 14))


CACHE = ROOT / "results" / "cache" / "shot_ext"


def train_job(job):
    run, n, readout, shots, f, seed = job
    cf = CACHE / f"{run}_{n}_{readout}_{shots}_{f}_{seed}.json"
    if cf.exists():                       # finished in an earlier, interrupted run
        d = json.loads(cf.read_text())
        return job, d["hist"], np.array(d["theta"]), d["seconds"]
    tr, te = FOLDS[f]
    transform = make_preprocessor(n)(X[tr], y[tr])
    clf = QuantumClassifier(n, depth_log(n), readout, loss="brier", shots=shots,
                            lr=LR_QUANTUM, epochs=SHOT_EPOCHS, batch_size=BATCH, seed=seed)
    t0 = time.time()
    hist = clf.fit(transform(X[tr]), y[tr], transform(X[te]), y[te])
    dt = time.time() - t0
    cf.write_text(json.dumps(dict(hist=hist, theta=clf.theta.tolist(), seconds=dt)))
    return job, hist, clf.theta, dt


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = [("C", n, s) for n in RUN_C_NS for s in (SHOTS_TRAIN, None)]
    cfg += [("D", RUN_D_N, s) for s in RUN_D_SHOTS]
    jobs = [(run, n, r, s, f, seed) for (run, n, s), r, f, seed in
            itertools.product(cfg, ("global", "local"), range(len(FOLDS)), INIT_SEEDS)]
    jobs.sort(key=lambda j: (j[3] is None, -j[1]))               # expensive first
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(train_job, jobs):
            results.append(res)
            (run, n, r, s, f, seed), hist, _, dt = res
            print(f"run {run} n={n} {r} shots={s} fold={f} seed={seed}: "
                  f"AUC={hist[-1]['test_auc']:.3f} ({dt:.0f}s, total {time.time() - t0:.0f}s)", flush=True)

    from qiskit_aer import AerSimulator
    from bp.qiskit_backend import sampled_scores
    sim = AerSimulator()
    curves, final = [], []
    for (run, n, readout, shots, f, seed), hist, theta, dt in results:
        mode = "exact" if shots is None else "shots"
        key = dict(run=run, n=n, readout=readout, mode=mode, shots_train=shots or 0, fold=f, seed=seed)
        curves += [dict(**key, **h) for h in hist]
        tr, te = FOLDS[f]
        Xte = make_preprocessor(n)(X[tr], y[tr])(X[te])
        s_shots = scores(y[te], sampled_scores(sim, n, depth_log(n), Xte, theta, readout,
                                               shots=SHOTS, seed=SEED + f))
        final.append(dict(**key, L=depth_log(n), seconds=dt,
                          init_exact_grad_norm=hist[0]["exact_grad_norm"],
                          signal_ratio=sum(h["exact_grad_norm"] for h in hist[1:])
                          / sum(h["grad_norm"] for h in hist[1:]),
                          **{f"exact_{k}": hist[-1][f"test_{k}"] for k in ("acc", "bacc", "auc", "loss")},
                          **{f"shots_{k}": v for k, v in s_shots.items()}))
    out = ROOT / "results"
    pd.DataFrame(curves).to_csv(out / "shot_ext_curves.csv", index=False)
    df = pd.DataFrame(final)
    df.to_csv(out / "shot_ext_final.csv", index=False)
    print(df.groupby(["run", "n", "shots_train", "readout"])[["exact_auc", "shots_auc", "signal_ratio"]]
          .agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
