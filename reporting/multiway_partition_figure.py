"""Multiway K^2-fold partition (Chiang et al. 2022) from DoubleML's multiway
example: (a) as DoubleML draws it, (b) clusters relabeled by fold -> blocks.
Blue = nuisance (train), red = score (test), white = unused.
Writes figures/<name>.pdf next to this script. 
Link: https://docs.doubleml.org/stable/examples/py_double_ml_multiway_cluster.html"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from doubleml import DoubleMLPLIV
from doubleml.plm.datasets import make_pliv_multiway_cluster_CKMS2021

np.random.seed(3141) # the notebook's seed
data = make_pliv_multiway_cluster_CKMS2021(25, 25, 100)
obj = DoubleMLPLIV(data, LassoCV(), LassoCV(), LassoCV(), n_folds=3)
K, splits = obj._n_folds_per_cluster, obj.smpls_cluster[0]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# A fold's test clusters are its members, so listing them fold by fold sorts
# the axis into blocks.
by_fold_row = np.concatenate([splits[i * K][1][0] for i in range(K)])
by_fold_col = np.concatenate([splits[j][1][1] for j in range(K)])


def draw(row_order, col_order, name):
    fig, axs = plt.subplots(K, K, figsize=(9, 9), constrained_layout=True)
    for p, ((train_r, train_c), (test_r, test_c)) in enumerate(splits):
        m = np.zeros((25, 25))
        m[np.ix_(train_r, train_c)] = -1               # nuisance (train)
        m[np.ix_(test_r, test_c)] = 1                  # score (test)
        ax = axs[p // K, p % K]
        ax.imshow(m[np.ix_(row_order, col_order)], cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        ax.set_title(f"Score: $I_{p // K + 1}$ × $J_{p % K + 1}$")
        ax.set(xticks=[], yticks=[])
    fig.savefig(OUT / f"{name}.pdf")                   # pdf: vector, best for Overleaf


draw(np.arange(25), np.arange(25), "multiway_partition_original")   # (a) as drawn
draw(by_fold_row, by_fold_col, "multiway_partition_blocks")         # (b) sorted by fold
