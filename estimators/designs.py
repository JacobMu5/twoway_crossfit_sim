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

Variables:
    DESIGNS (dict): maps design names to fold functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np

MULTIWAY_K = 2
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
