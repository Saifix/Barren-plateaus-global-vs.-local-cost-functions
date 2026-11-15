"""Classical baselines on the committed folds, same preprocessing as the quantum model.

results/baselines.csv     one row per (n, model, fold, seed)
results/mlp_curves.csv    per-epoch curves of the parameter-matched MLP
"""
import pandas as pd

from bp.ansatz import depth_log
from bp.classical import matched_hidden, mlp_params, mlp_with_curves, run_baseline
from bp.config import CLF_NS, EPOCHS, BATCH, INIT_SEEDS, LR_MLP
from bp.data import ROOT, load_data, load_folds, make_preprocessor

X, y = load_data()
folds = load_folds()
rows, curves = [], []
for n in CLF_NS:
    L = depth_log(n)
    h = matched_hidden(n, 2 * n * L)
    for f, (tr, te) in enumerate(folds):
        transform = make_preprocessor(n)(X[tr], y[tr])
        Xtr, Xte = transform(X[tr]), transform(X[te])
        for model in ("majority", "logreg", "svm_rbf"):
            rows.append(dict(n=n, model=model, fold=f, seed=0,
                             **run_baseline(model, Xtr, y[tr], Xte, y[te])))
        for seed in INIT_SEEDS:
            m, hist = mlp_with_curves(Xtr, y[tr], Xte, y[te], hidden=h, epochs=EPOCHS,
                                      lr=LR_MLP, batch_size=BATCH, seed=seed)
            last = hist[-1]
            rows.append(dict(n=n, model="mlp", fold=f, seed=seed, acc=last["test_acc"],
                             bacc=last["test_bacc"], auc=last["test_auc"],
                             hidden=h, params=mlp_params(n, h)))
            curves += [dict(n=n, fold=f, seed=seed, **r) for r in hist]
        rows[-1]["pca_var"] = transform.explained_variance
    print(f"n={n}: MLP hidden={h} ({mlp_params(n, h)} params vs quantum {2 * n * L})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "baselines.csv", index=False)
pd.DataFrame(curves).to_csv(ROOT / "results" / "mlp_curves.csv", index=False)
print(df.groupby(["n", "model"])[["acc", "bacc", "auc"]].agg(["mean", "std"]).round(3))
