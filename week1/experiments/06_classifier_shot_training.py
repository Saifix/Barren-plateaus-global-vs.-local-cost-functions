"""H4, the test that can actually show a barren plateau: training under shot noise.

Brier loss, n in SHOT_NS, global vs. local readout, 5 folds x 3 init seeds, in
two gradient modes on identical folds/seeds/hyper-parameters:
  exact  adjoint gradients (control)
  shots  parameter-shift gradients, every circuit estimated from SHOTS_TRAIN samples
Every trained model is then evaluated on its test fold exactly and with SHOTS
samples on Aer. Primary metric: test ROC-AUC (threshold-free).

results/shot_curves.csv, results/shot_final.csv
"""
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from bp import SEED
from bp.ansatz import depth_log
from bp.config import BATCH, INIT_SEEDS, LR_QUANTUM, SHOT_EPOCHS, SHOT_NS, SHOTS, SHOTS_TRAIN
from bp.data import load_data, load_folds, make_preprocessor, results_dir
from bp.metrics import scores
from bp.qclassifier import QuantumClassifier

X, y = load_data()
FOLDS = load_folds()
MAX_WORKERS = 10


def train_job(job):
    n, readout, mode, f, seed = job
    tr, te = FOLDS[f]
    transform = make_preprocessor(n)(X[tr], y[tr])
    clf = QuantumClassifier(n, depth_log(n), readout, loss="brier",
                            shots=SHOTS_TRAIN if mode == "shots" else None,
                            lr=LR_QUANTUM, epochs=SHOT_EPOCHS, batch_size=BATCH, seed=seed)
    hist = clf.fit(transform(X[tr]), y[tr], transform(X[te]), y[te])
    return job, hist, clf.theta


def main():
    jobs = list(itertools.product(SHOT_NS, ("global", "local"), ("exact", "shots"),
                                  range(len(FOLDS)), INIT_SEEDS))
    jobs.sort(key=lambda j: (j[2] != "shots", -j[0]))        # expensive jobs first
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(train_job, jobs))
    print(f"trained {len(results)} models in {time.time() - t0:.0f}s", flush=True)

    from qiskit_aer import AerSimulator
    from bp.qiskit_backend import sampled_scores
    sim = AerSimulator()
    curves, final = [], []
    for (n, readout, mode, f, seed), hist, theta in results:
        key = dict(n=n, readout=readout, mode=mode, fold=f, seed=seed)
        curves += [dict(**key, **h) for h in hist]
        tr, te = FOLDS[f]
        Xte = make_preprocessor(n)(X[tr], y[tr])(X[te])
        s_shots = scores(y[te], sampled_scores(sim, n, depth_log(n), Xte, theta, readout,
                                               shots=SHOTS, seed=SEED + f))
        final.append(dict(**key, L=depth_log(n),
                          init_exact_grad_norm=hist[0]["exact_grad_norm"],
                          **{f"exact_{k}": hist[-1][f"test_{k}"] for k in ("acc", "bacc", "auc", "loss")},
                          **{f"shots_{k}": v for k, v in s_shots.items()}))
    out = results_dir(1)
    pd.DataFrame(curves).to_csv(out / "shot_curves.csv", index=False)
    df = pd.DataFrame(final)
    df.to_csv(out / "shot_final.csv", index=False)
    print(df.groupby(["mode", "n", "readout"])[["exact_auc", "exact_bacc", "shots_auc"]]
          .agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
