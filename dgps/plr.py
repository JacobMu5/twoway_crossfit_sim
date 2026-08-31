"""
Module for the shared data structure of the two-way clustered PLR DGPs.

All DGPs draw data on an N x M grid of cells from the partially linear
model
    D_ij = m0(X_ij) + V_ij
    Y_ij = theta0 * D_ij + g0(X_ij) + eps_ij
where the covariates, V, and eps each have a row effect, a column
effect, and a cell-specific part. The true nuisance functions come back
with every sample, which is what allows the bias dissection. The DGP
classes live in dgps.reveal_plr (the paper DGP) and dgps.legacy_plr
(superseded DGPs kept for reference; not used by any campaign).

Classes:
    ClusteredSample: one realised data set with the true nuisances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusteredSample:
    """One realised data set with the true nuisances.

    Holds the data (x, d, y), the grid layout (rows, cols, n_rows,
    n_cols), the truth (theta0, m0, l0), the residuals (v, eps), and
    the row/column/cell parts of v (they sum to v).
    """

    x: np.ndarray
    d: np.ndarray
    y: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    n_rows: int
    n_cols: int
    theta0: float
    m0: np.ndarray
    l0: np.ndarray
    v: np.ndarray
    eps: np.ndarray
    v_row: np.ndarray
    v_col: np.ndarray
    v_cell: np.ndarray
