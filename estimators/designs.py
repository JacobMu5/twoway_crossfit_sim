"""
Module for the cross-fitting designs.

Each design is a function (n_rows, n_cols, rng) -> list[Split] that says
which cells to train on and which cells to predict for. The designs:

* no_cf: no cross-fitting; train and predict on everything.
* as_iid: ordinary K-fold over cells, ignoring the clustering.
* as_iid_matched: as-IID folds with training thinned to n/4 cells, to
  match the multiway training size.
* multiway: the K^2-fold scheme of Chiang, Kato, Ma & Sasaki (2022):
  hold out a row block and a column block together, so no training cell
  shares a row or column with a predicted cell.
* cluster_oob_sub: a sub-sampled two-way cluster bootstrap forest where
  each bag trains on a random ceil(N^0.45) x ceil(M^0.45) block of
  clusters and each cell is predicted only by bags that drew neither its
  row cluster nor its column cluster

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


def matched_cell_folds(
    n_rows: int, n_cols: int, rng: np.random.Generator
) -> list[Split]:
    """K=2 cell folds with training sets thinned to n/4 cells.

    This matches the multiway training size, so the as-IID vs multiway
    comparison isolates fold geometry from data volume.
    """
    n = n_rows * n_cols
    if n < 40:
        raise ValueError(
            f"matched design needs n >= 40 cells for the n/4 thinning, got {n}"
        )
    return [Split(rng.choice(sp.train, size=round(n / 4), replace=False), sp.pred)
            for sp in cell_folds(n_rows, n_cols, rng)]


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
    "as_iid_matched": matched_cell_folds,
    "multiway": multiway_folds,
}

def predict_cluster_oob_sub(
    x, target, rows, cols, n_rows, n_cols, learner: LearnerSpec, seed,
    n_bags: int | None = None, min_cells: int = 10, min_trees: int = 8,
    return_stats: bool = False, clip_predictions: bool = True,
    sub_exponent: float = SUB_EXPONENT,
):
    """Sub-sampled two-way cluster forest with leave-row-and-column-out OOB.

    Each of B bags trains one base learner on a random block of
    ceil(N**gamma) rows x ceil(M**gamma) columns drawn without
    replacement. A cell is then predicted by averaging only the bags that
    drew neither its row cluster nor its column cluster, so no bag ever
    saw a unit correlated with the target cell. Per-bag predictions are
    clipped to the in-bag target range.

    Args:
        n_bags (int | None): number of bag draws B; defaults to the
            learner's n_bags.
        min_cells (int): minimum trained cells required to keep a bag.
        min_trees (int): minimum omit-both bags required per cell.
        return_stats (bool): if True, also return the bag-count statistics.
        clip_predictions (bool): clip per-bag predictions to the in-bag range.
        sub_exponent (float): subsampling exponent gamma.

    Returns:
        np.ndarray: out-of-bag predictions, and a stats dict if requested.

    Raises:
        RuntimeError: if any cell has fewer than min_trees omit-both bags.
    """
    m_r = max(2, int(np.ceil(n_rows ** sub_exponent)))
    m_c = max(2, int(np.ceil(n_cols ** sub_exponent)))
    B = learner.n_bags if n_bags is None else n_bags
    rng = np.random.default_rng(seed)
    n = len(target)
    preds = np.zeros((B, n))
    used = np.zeros(B, dtype=bool)
    omit_row = np.ones((B, n_rows), dtype=bool)
    omit_col = np.ones((B, n_cols), dtype=bool)

    # Fit one base learner per bag on its drawn row x column block
    for b in range(B):
        rin = rng.choice(n_rows, size=m_r, replace=False)
        cin = rng.choice(n_cols, size=m_c, replace=False)
        omit_row[b, rin] = False
        omit_col[b, cin] = False
        row_mask = np.zeros(n_rows, dtype=bool)
        row_mask[rin] = True
        col_mask = np.zeros(n_cols, dtype=bool)
        col_mask[cin] = True
        in_bag = row_mask[rows] & col_mask[cols]
        if in_bag.sum() < min_cells:
            continue
        model = learner.base(seed + b)
        model.fit(x[in_bag], target[in_bag])
        p = np.asarray(model.predict(x))
        if clip_predictions:
            p = np.clip(p, float(target[in_bag].min()), float(target[in_bag].max()))
        preds[b] = p
        used[b] = True

    # Average, per cell, only the bags omitting both its row and column
    out = np.empty(n)
    both_counts = np.empty(n, dtype=int)
    for idx in range(n):
        both = used & omit_row[:, rows[idx]] & omit_col[:, cols[idx]]
        both_counts[idx] = both.sum()
        if both.sum() < min_trees:
            raise RuntimeError(
                f"cluster_oob_sub: only {int(both.sum())} omit-both bags for "
                f"cell {idx} (min_trees={min_trees}); raise n_bags -- never "
                "fall back to dishonest bags (audit repair)"
            )
        out[idx] = preds[both, idx].mean()
    if return_stats:
        stats = {
            "mean_both_trees": float(both_counts.mean()),
            "min_both_trees": float(both_counts.min()),
        }
        return out, stats
    return out


# Bagged designs produce predictions directly (a cell is an average over
# its omit-both bags), so they cannot be expressed as list[Split] and
# live in their own registry; PLRDMLEstimator dispatches on membership.
BAGGED_DESIGNS: dict[str, Callable] = {
    "cluster_oob_sub": predict_cluster_oob_sub,
}