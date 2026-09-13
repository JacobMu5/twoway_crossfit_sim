"""
Module for the cross-fitting designs.

Each design is a function (n_rows, n_cols, rng) -> list[Split] that says
which cells to train on and which cells to predict for. The designs:

* no_cf: no cross-fitting; train and predict on everything.
* as_iid: ordinary K-fold over cells, ignoring the clustering.
* multiway: the K^2-fold scheme of Chiang, Kato, Ma & Sasaki (2022):
  hold out a row block and a column block together, so no training cell
  shares a row or column with a predicted cell.
* cluster_oob_sub: a sub-sampled two-way cluster bootstrap forest where
  each bag trains on a random ceil(N^0.45) x ceil(M^0.45) block of
  clusters and each cell is predicted by exactly K bags that drew neither
  its row cluster nor its column cluster
* cluster_oob_nodrop: the no-drop version of cluster_oob_sub (the
  subsampled no-drop variant discussed by Chen & Chiang 2026): identical
  bag draws and the same K bags per cell, but bags sharing the cell's row or
  column cluster (its "cross") are not dropped.

Variables:
    DESIGNS (dict): maps design names to fold functions.
    BAGGED_DESIGNS (dict): maps design names to prediction functions.
    SUB_EXPONENT (float): default subsampling exponent gamma.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np

from estimators.learners import LearnerSpec

MULTIWAY_K = 2
SUB_EXPONENT = 0.45
# Seed contract: the runner passes first_seed + sim_id to dgp.sample() and

EST_SEED_OFFSET = 7_000_003


@dataclass(frozen=True)
class Split:
    """A single cross-fitting split: indices to train on and to predict on."""

    train: np.ndarray
    pred: np.ndarray


def full_split(n_rows: int, n_cols: int, rng: np.random.Generator) -> list[Split]:
    """Degenerate split used by no-CF: train and predict on the whole sample."""
    idx = np.arange(n_rows * n_cols)
    return [Split(idx, idx)]


def cell_folds(
    n_rows: int, n_cols: int, rng: np.random.Generator, k: int = 2
) -> list[Split]:
    """Ordinary K-fold split of cells that ignores the cluster structure."""
    order = rng.permutation(n_rows * n_cols)
    return [Split(np.setdiff1d(order, fold), fold)
            for fold in np.array_split(order, k)]


def multiway_folds(
    n_rows: int, n_cols: int, rng: np.random.Generator, k: int = MULTIWAY_K
) -> list[Split]:
    """K^2-fold two-way split: hold out a row block and a column block.

    Each fold predicts the cells in one (row block, column block) pair
    and trains on the cells sharing neither.
    """
    idx = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    row_blocks = [np.sort(b) for b in np.array_split(rng.permutation(n_rows), k)]
    col_blocks = [np.sort(b) for b in np.array_split(rng.permutation(n_cols), k)]
    out: list[Split] = []
    for rb, cb in product(row_blocks, col_blocks):
        pred = idx[np.ix_(rb, cb)].ravel()
        train = idx[np.ix_(np.setdiff1d(np.arange(n_rows), rb),
                           np.setdiff1d(np.arange(n_cols), cb))].ravel()
        if pred.size:
            out.append(Split(train, pred))
    return out


def require_canonical_grid(rows, cols, n_rows, n_cols) -> None:
    """Raise an error if the sample is not in canonical N x M grid order.

    multiway_folds builds its folds from n_rows and n_cols alone, so on
    shuffled data it would silently produce leaky folds.
    """
    if not (np.array_equal(rows, np.repeat(np.arange(n_rows), n_cols))
            and np.array_equal(cols, np.tile(np.arange(n_cols), n_rows))):
        raise ValueError(
            "multiway design assumes the canonical N x M grid layout "
            "(rows = repeat(arange(N), M), cols = tile(arange(M), N))"
        )


DESIGNS: dict[str, Callable[..., list[Split]]] = {
    "no_cf": full_split,
    "as_iid": cell_folds,
    "multiway": multiway_folds,
}

def _fit_bags(
    x, target, rows, cols, n_rows, n_cols, learner: LearnerSpec, seed,
    n_bags: int | None, min_cells: int, sub_exponent: float,
):
    """Fit one base learner per sub-sampled cluster bag; shared bag loop.

    Bags are drawn until every cell has K = n_bags bags avoiding its row and
    column. Each bag trains on a random ceil(N**gamma) x ceil(M**gamma) block
    of clusters drawn without replacement and predicts every cell. Both bagged
    designs draw the same bags from the same seed and differ only in how they
    average them, so this loop is written once. Returns the per-bag
    predictions and the per-bag row/column omission masks.
    """
    m_r = max(2, int(np.ceil(n_rows ** sub_exponent)))
    m_c = max(2, int(np.ceil(n_cols ** sub_exponent)))
    if m_r * m_c < min_cells:
        raise ValueError(f"bags of {m_r} x {m_c} cells are below the {min_cells}-cell minimum")
    K = learner.n_bags if n_bags is None else n_bags
    rng = np.random.default_rng(seed)
    n = len(target)
     
    # Draw row x column blocks until every cell has K bags avoiding its cross
    blocks, eligible = [], np.zeros(n, dtype=int)
    while eligible.min() < K:
        rin = rng.choice(n_rows, size=m_r, replace=False)
        cin = rng.choice(n_cols, size=m_c, replace=False)
        blocks.append((rin, cin))
        eligible += ~np.isin(rows, rin) & ~np.isin(cols, cin)
    B = len(blocks)
    preds = np.zeros((B, n))
    used = np.zeros(B, dtype=bool)
    omit_row = np.ones((B, n_rows), dtype=bool)
    omit_col = np.ones((B, n_cols), dtype=bool)

    # Fit one base learner per bag on its drawn row x column block
    for b, (rin, cin) in enumerate(blocks):
        omit_row[b, rin] = False
        omit_col[b, cin] = False
        row_mask = np.zeros(n_rows, dtype=bool)
        row_mask[rin] = True
        col_mask = np.zeros(n_cols, dtype=bool)
        col_mask[cin] = True
        in_bag = row_mask[rows] & col_mask[cols]
        model = learner.make(seed + b)
        model.fit(x[in_bag], target[in_bag])
        preds[b] = np.asarray(model.predict(x))
    return preds, omit_row, omit_col


def predict_cluster_oob_sub(
    x, target, rows, cols, n_rows, n_cols, learner: LearnerSpec, seed,
    n_bags: int | None = None, min_cells: int = 10, return_stats: bool = False,
    sub_exponent: float = SUB_EXPONENT,
):
    """Sub-sampled two-way cluster forest with leave-row-and-column-out OOB.

    Each of B bags trains one base learner on a random block of
    ceil(N**gamma) rows x ceil(M**gamma) columns drawn without
    replacement. A cell is then predicted by averaging exactly K = n_bags bags
    that drew neither its row cluster nor its column cluster, so no bag ever
    saw a unit correlated with the target cell.

    Args:
        n_bags (int | None): bags averaged at every cell, K; defaults to the
            learner's n_bags.
        min_cells (int): minimum number of cells a bag must train on.
        return_stats (bool): if True, also return the bag-count statistics.
        sub_exponent (float): subsampling exponent gamma.

    Returns:
        np.ndarray: out-of-bag predictions, and a stats dict if requested.
    """
    preds, omit_row, omit_col = _fit_bags(
        x, target, rows, cols, n_rows, n_cols, learner, seed,
        n_bags, min_cells, sub_exponent,
    )
    K = learner.n_bags if n_bags is None else n_bags
    n = len(target)

    # Average, per cell, its first K bags omitting both its row and column
    out = np.empty(n)
    for idx in range(n):
        both = np.flatnonzero(omit_row[:, rows[idx]] & omit_col[:, cols[idx]])[:K]
        out[idx] = preds[both, idx].mean()
    if return_stats:
        return out, {"bags_per_cell": float(K), "bags_drawn": float(len(preds))}
    return out

def predict_cluster_nodrop(
    x, target, rows, cols, n_rows, n_cols, learner: LearnerSpec, seed,
    n_bags: int | None = None, min_cells: int = 10, return_stats: bool = False,
    sub_exponent: float = SUB_EXPONENT,
):
    """No-drop comparison baseline: the same bags, averaged WITHOUT the drop.

    Identical to predict_cluster_oob_sub -- same block sizes, same bag draws
    from the same seed, same K bags per cell -- except every cell averages the
    first K fitted bags whether or not they trained on its own row or column
    cluster. A cell is therefore predicted partly by bags that
    trained on its own row or column cluster, so the fitted nuisance error is
    correlated with the cell's own cluster shock (the "leak"). Running this next
    to predict_cluster_oob_sub with common draws isolates exactly what the
    cross-drop buys: honesty (a structurally mean-zero own-cluster term) versus
    an equal-or-larger bag average whose validity rests on that leak cancelling.

    Returns:
        np.ndarray: predictions, and a stats dict if requested.
    """
    preds, _, _ = _fit_bags(
        x, target, rows, cols, n_rows, n_cols, learner, seed,
        n_bags, min_cells, sub_exponent,
    )
    K = learner.n_bags if n_bags is None else n_bags
    out = preds[:K].mean(axis=0)
    if return_stats:
        return out, {"bags_per_cell": float(K), "bags_drawn": float(len(preds))}
    return out

# Bagged designs produce predictions directly (a cell is an average over
# its omit-both bags), so they cannot be expressed as list[Split] and
# live in their own registry; PLRDMLEstimator dispatches on membership.
# cluster_oob_nodrop is the honesty-off twin of cluster_oob_sub (see above).
BAGGED_DESIGNS: dict[str, Callable] = {
    "cluster_oob_sub": predict_cluster_oob_sub,
    "cluster_oob_nodrop": predict_cluster_nodrop,
}