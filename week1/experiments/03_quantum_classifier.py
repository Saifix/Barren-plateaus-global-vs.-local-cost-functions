"""First quantum model: global vs. local readout, same circuit, same folds (H4 preview).

1. Train (exact gradients) for every n x readout x fold x init seed.
2. Re-evaluate every trained model on the test fold with SHOTS samples on
   AerSimulator, as NB11 requires (reported next to the exact numbers).
3. Label-shuffle leakage check.

results/qclf_curves.csv, results/qclf_final.csv, results/leakage_check.csv
"""
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bp import SEED
from bp.ansatz import depth_log
from bp.config import BATCH, CLF_NS, EPOCHS, INIT_SEEDS, LR_QUANTUM, SHOTS
from bp.data import load_data, load_folds, make_preprocessor, results_dir
from bp.metrics import scores
from bp.qclassifier import QuantumClassifier

# Qiskit is imported in main() only, so the training workers stay small.
MAX_WORKERS = 8

X, y = load_data()
FOLDS = load_folds()


def train_job(job):
    n, readout, f, seed, shuffle = job
    tr, te = FOLDS[f]
    ytr = y[tr]
    if shuffle:
        ytr = np.random.default_rng([SEED, f]).permutation(ytr)
    transform = make_preprocessor(n)(X[tr], ytr)
    clf = QuantumClassifier(n, depth_log(n), readout, lr=LR_QUANTUM, epochs=EPOCHS,
                            batch_size=BATCH, seed=seed)
    hist = clf.fit(transform(X[tr]), ytr, transform(X[te]), y[te])
    return job, hist, clf.theta


def main():
    jobs = [(n, r, f, s, False) for n, r, f, s in
            itertools.product(CLF_NS, ("global", "local"), range(len(FOLDS)), INIT_SEEDS)]
    leak_jobs = [(4, r, f, SEED, True) for r in ("global", "local") for f in range(len(FOLDS))]
    t0 = time.time()
    # largest jobs first so the pool drains evenly
    order = sorted(jobs + leak_jobs, key=lambda j: -j[0])
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(train_job, order))
    print(f"trained {len(results)} models in {time.time() - t0:.0f}s", flush=True)

    from qiskit_aer import AerSimulator
    from bp.qiskit_backend import sampled_scores

    curves, final, leak = [], [], []
    sim = AerSimulator()
    for (n, readout, f, seed, shuffle), hist, theta in results:
        key = dict(n=n, readout=readout, fold=f, seed=seed)
        if shuffle:
            leak.append(dict(model=f"quantum-{readout}", n=n, fold=f,
                             acc=hist[-1]["test_acc"], auc=hist[-1]["test_auc"]))
            continue
        curves += [dict(**key, **h) for h in hist]
        tr, te = FOLDS[f]
        Xte = make_preprocessor(n)(X[tr], y[tr])(X[te])
        p_shots = sampled_scores(sim, n, depth_log(n), Xte, theta, readout,
                                 shots=SHOTS, seed=SEED + f)
        s_shots = scores(y[te], p_shots)
        final.append(dict(**key, L=depth_log(n), params=len(theta),
                          init_grad_norm=hist[0]["grad_norm"],
                          **{f"exact_{k}": hist[-1][f"test_{k}"] for k in ("acc", "bacc", "auc", "loss")},
                          **{f"shots_{k}": v for k, v in s_shots.items()}))
    print(f"Aer evaluation done at {time.time() - t0:.0f}s", flush=True)

    # (the classical leakage reference, with many permutations, is 05_leakage_check.py)
    out = results_dir(1)
    pd.DataFrame(curves).to_csv(out / "qclf_curves.csv", index=False)
    df = pd.DataFrame(final)
    df.to_csv(out / "qclf_final.csv", index=False)
    pd.DataFrame(leak).to_csv(out / "leakage_check.csv", index=False)
    cols = ["exact_acc", "exact_bacc", "exact_auc", "shots_acc", "shots_auc", "init_grad_norm"]
    print(df.groupby(["n", "readout"])[cols].mean().round(3))
    print(pd.DataFrame(leak).groupby("model")[["acc", "auc"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
