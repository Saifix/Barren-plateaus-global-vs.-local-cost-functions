"""Label-shuffle control for the classical pipeline, with many permutations.

Under shuffled training labels the held-out AUC is extremely spread out
(s.d. ~0.27): the first principal component is so predictive that a model
fitted to random labels lands near 0.15 or 0.85 depending on a random sign.
Five permutations (one per fold) therefore cannot distinguish 'chance' from
'leakage'; 40 per fold can.  -> results/leakage_check_logreg.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from bp import SEED
from bp.data import load_data, load_folds, make_preprocessor, results_dir

PERMS_PER_FOLD = 40
X, y = load_data()
rows = []
for f, (tr, te) in enumerate(load_folds()):
    for r in range(PERMS_PER_FOLD):
        ytr = np.random.default_rng([SEED, f, r]).permutation(y[tr])
        tf = make_preprocessor(4)(X[tr], ytr)
        m = LogisticRegression(max_iter=5000).fit(tf(X[tr]), ytr)
        rows.append(dict(model="logreg", n=4, fold=f, perm=r,
                         acc=accuracy_score(y[te], m.predict(tf(X[te]))),
                         auc=roc_auc_score(y[te], m.predict_proba(tf(X[te]))[:, 1])))
df = pd.DataFrame(rows)
df.to_csv(results_dir(1) / "leakage_check_logreg.csv", index=False)
se = df.auc.std() / np.sqrt(len(df))
print(f"{len(df)} permutations: acc {df.acc.mean():.3f}, AUC {df.auc.mean():.3f} ± {se:.3f} (s.e.)")
