"""Create the fold object ONCE and commit it. Every table uses data/folds.json."""
import numpy as np

from bp.data import FOLDS_PATH, load_data, make_folds, save_folds

X, y = load_data()
folds = make_folds(y)
save_folds(folds, y)
print(f"X {X.shape}  positives (malignant) {np.sum(y == 1)}  negatives {np.sum(y == -1)}")
print(f"majority-class rate: {max(np.mean(y == 1), np.mean(y == -1)):.3f}")
print("test sizes:", [len(te) for _, te in folds], "->", FOLDS_PATH)
