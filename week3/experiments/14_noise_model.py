"""Q2: an analytical model of how depth and noise change the global/local gap.

bp.pauli_path computes the gradient variance of the random-angle HEA exactly
(as an average over uniform angles, no sampling), with depolarizing noise
included exactly and amplitude damping through its unital part only.
  A  validation against every simulation we have:
       Week 1 noiseless sweep (n <= 12; n = 14 needs 4^14 doubles), and the
       Week 2 density-matrix study (n-, depth- and strength-studies), for
       var_k (theta_1) and var_mean (average over parameters);
  B  regime map beyond what density matrices can reach: theta_1 variance for
     n in MODEL_NS, L in MODEL_DEPTHS, depolarizing p2 in {0} + MODEL_P2
     (one-qubit strength p2 / 10, as in Week 2);
  C  the mechanism: average Pauli weight of the backward-propagated observable,
     layer by layer (the local cost's is capped by its light cone).

results/noise_model_validation.csv, noise_model_map.csv, noise_model_weights.csv
"""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bp.ansatz import depth_log
from bp.config import MODEL_DEPTHS, MODEL_NS, MODEL_P2, VAR_K_FIXED
from bp.data import results_dir
from bp.pauli_path import PathNoise, mean_pauli_weight, variance_summary

MAX_WORKERS = int(os.environ.get("BP_WORKERS", 3))


def path_noise(model, p2):
    if model == "depolarizing":
        return PathNoise(p1=p2 / 10, p2=p2)
    if model == "amplitude_damping":
        return PathNoise(g1=p2 / 10, g2=p2)
    return PathNoise()


def validate_week1():
    sim = pd.read_csv(results_dir(1) / "variance_sweep.csv")
    sim = sim[sim.n <= 12]
    rows = []
    for _, r in sim.iterrows():
        m = variance_summary(int(r.n), int(r.L), r.cost, all_params=False, k_fixed=VAR_K_FIXED)
        rows.append(dict(source="week1_sweep", study="noiseless", n=r.n, L=r.L, model="noiseless", p2=0.0,
                         cost=r.cost, stat="var_k", sim=r.var_k, sim_lo=r.var_k_lo, sim_hi=r.var_k_hi,
                         model_value=m["var_k"]))
    return rows


def validate_week2_row(r):
    m = variance_summary(int(r["n"]), int(r["L"]), r["cost"], path_noise(r["model"], r["p2"]),
                         k_fixed=VAR_K_FIXED)
    return [dict(source="week2_density", study=r["study"], n=r["n"], L=r["L"], model=r["model"],
                 p2=r["p2"], cost=r["cost"], stat=s, sim=r[s], sim_lo=r[f"{s}_lo"], sim_hi=r[f"{s}_hi"],
                 model_value=m[s]) for s in ("var_k", "var_mean")]


def map_job(job):
    n, L, p2, cost = job
    v = variance_summary(n, L, cost, PathNoise(p1=p2 / 10, p2=p2), all_params=False)["var_k"]
    return dict(n=n, L=L, p2=p2, cost=cost, var_k=v)


def main():
    out = results_dir(3)
    sim2 = pd.read_csv(results_dir(2) / "noise_variance.csv").to_dict("records")
    jobs = [(n, L, p2, c) for n, L, p2, c in
            itertools.product(MODEL_NS, MODEL_DEPTHS, [0.0] + MODEL_P2, ("global", "local"))]
    jobs.sort(key=lambda j: -(4 ** j[0]) * j[1])
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        val = [r for rows in ex.map(validate_week2_row, sim2) for r in rows]
        val += validate_week1()
        v = pd.DataFrame(val)
        v["ratio"] = v.model_value / v.sim
        v["inside_ci"] = (v.model_value >= v.sim_lo) & (v.model_value <= v.sim_hi)
        v.to_csv(out / "noise_model_validation.csv", index=False)
        print(v.groupby(["source", "model"]).agg(rows=("ratio", "size"), inside_ci=("inside_ci", "mean"),
                                                 median_ratio=("ratio", "median")).round(3).to_string(),
              flush=True)
        mp = pd.DataFrame(list(ex.map(map_job, jobs)))
    mp.to_csv(out / "noise_model_map.csv", index=False)
    w = [dict(n=n, L=L, cost=c, layer_from_end=i + 1, mean_weight=x)
         for n, L in ((8, 8), (10, depth_log(10)), (10, 10)) for c in ("global", "local")
         for i, x in enumerate(mean_pauli_weight(n, L, c))]
    pd.DataFrame(w).to_csv(out / "noise_model_weights.csv", index=False)
    print("wrote noise_model_validation.csv, noise_model_map.csv, noise_model_weights.csv")


if __name__ == "__main__":
    main()
