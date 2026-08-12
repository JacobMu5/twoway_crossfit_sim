"""
Module for the Balkus, Laith and Hejazi (2026) two-way clustered DGP.

Port of the authors' hard_cluster.R simulation DGP.
Row and column shocks enter the covariates, the treatment, and the
outcome, so Y is strongly two-way clustered, but the oracle AIPW
influence function carries almost none of that clustering, which is
what the Hoeffding diagnostic measures.

Functions:
    draw_cells: sample cells and the oracle influence functions.
    sample_grid: sample one full N x M grid with shared shocks.
    make_row_latents: row latents for the conditional method.
    analyst_propensity: propensity with cluster shocks integrated out.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

_GH_X, _GH_W = hermegauss(40)
_GH_W = _GH_W / _GH_W.sum()
_SD_U = math.sqrt(2 * 0.6**2)

TRUE_ATE: float = 1.045  # E[tau] = 1 + 0.3 E[L3 L4] = 1 + 0.3 * 0.15


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-z))


def analyst_propensity(h: np.ndarray) -> np.ndarray:
    """Propensity with the assignment cluster shocks integrated out.

    Args:
        h (np.ndarray): the covariate index of the assignment equation.

    Returns:
        np.ndarray: E over the cluster shocks of sigmoid((h + shocks) / 5).
    """
    return _sigmoid((h[..., None] + _SD_U * _GH_X) / 5.0) @ _GH_W


def draw_cells(
    n: int,
    row_lat: dict | None,
    rng: np.random.Generator,
    theta_w: float | None = None,
) -> dict:
    """Sample one block of cells from the hard_cluster.R DGP.

    Column and cell shocks are always new drwas; row latents can be frozen
    so the conditional method can estimate Var(E[f | row latents]).
    Besides the outcome and the oracle AIPW influence function, the
    sampler returns the oracle PLR score once theta_w is given.

    Args:
        n (int): number of cells to draw.
        row_lat (dict | None): frozen row latents of length n, or None
            for fresh latents per cell.
        rng (np.random.Generator): source of randomness.
        theta_w (float | None): calibrated PLR estimand; while None the
            PLR score is omitted. Defaults to None.

    Returns:
        dict: outcome "Y", AIPW influence function "phi", and (once
        theta_w is given) the PLR score "phi_plr".
    """
    if row_lat is None:
        a_x, a_m, a_t, a_0 = (rng.standard_normal(n) for _ in range(4))
    else:
        a_x, a_m, a_t, a_0 = (row_lat[k] for k in ("x", "m", "t", "0"))
    b_x, b_m, b_t, b_0 = (rng.standard_normal(n) for _ in range(4))

    # Covariates (their L1..L5)
    l1 = a_x + b_x + rng.standard_normal(n)
    l2 = rng.standard_normal(n)
    l3 = np.sin(l1) + 0.3 * l2 + rng.normal(0.0, 0.5, n)
    l4 = (l1 > 0) * l2 + rng.normal(0.0, 0.5, n)
    l5 = rng.normal(1.0, 2.0, n)

    # Assignment and outcome equations
    h = -0.4 + 0.8 * l1 - 0.7 * l2**2 + 0.5 * np.sin(l3) + 0.4 * l1 * l2 - 0.5 * l4 * l5
    q = _sigmoid((h + 0.6 * a_m - 0.6 * b_m) / 5.0)
    treat = (rng.random(n) < q).astype(float)

    tau = 1.0 + 0.5 * np.sin(l1) - 0.5 * np.sin(l2) + 0.3 * l3 * l4
    base = 2.0 + 0.5 * l1 - 0.4 * l2 + 0.3 * l3**2 - 0.5 * np.sin(l4) + 0.4 * l1 * l2
    y = (base + 0.6 * a_0 + 0.6 * b_0
         + treat * (tau + 0.4 * a_t * b_t) + rng.normal(0.0, 0.5, n))

    # Oracle scores at the cluster-blind nuisances
    g = analyst_propensity(h)
    m1, m0 = base + tau, base
    phi = (m1 - m0) + treat * (y - m1) / g - (1 - treat) * (y - m0) / (1 - g)
    out = {"Y": y, "phi": phi}
    if theta_w is not None:
        vtil = treat - g
        ytil = y - (base + tau * g)
        out["phi_plr"] = vtil * (ytil - theta_w * vtil)
    return out


def sample_grid(n_rows: int, n_cols: int, rng: np.random.Generator) -> dict:
    """Sample one full N x M grid with shared row and column shocks.

    Unlike draw_cells, the row shocks are shared by all cells of a row
    and the column shocks by all cells of a column, exactly as in the
    authors' hard_cluster.R.

    Args:
        n_rows (int): number of row clusters N.
        n_cols (int): number of column clusters M.
        rng (np.random.Generator): source of randomness.

    Returns:
        dict: covariates "L", treatment "A", outcome "Y", the analyst
        nuisances "g", "m1", "m0", and the indices "rows", "cols".
    """
    a_x, a_m, a_t, a_0 = rng.standard_normal((4, n_rows))
    b_x, b_m, b_t, b_0 = rng.standard_normal((4, n_cols))
    rows = np.repeat(np.arange(n_rows), n_cols)
    cols = np.tile(np.arange(n_cols), n_rows)
    n = n_rows * n_cols

    l1 = a_x[rows] + b_x[cols] + rng.standard_normal(n)
    l2 = rng.standard_normal(n)
    l3 = np.sin(l1) + 0.3 * l2 + rng.normal(0.0, 0.5, n)
    l4 = (l1 > 0) * l2 + rng.normal(0.0, 0.5, n)
    l5 = rng.normal(1.0, 2.0, n)

    h = -0.4 + 0.8 * l1 - 0.7 * l2**2 + 0.5 * np.sin(l3) + 0.4 * l1 * l2 - 0.5 * l4 * l5
    q = _sigmoid((h + 0.6 * a_m[rows] - 0.6 * b_m[cols]) / 5.0)
    treat = (rng.random(n) < q).astype(float)

    tau = 1.0 + 0.5 * np.sin(l1) - 0.5 * np.sin(l2) + 0.3 * l3 * l4
    base = 2.0 + 0.5 * l1 - 0.4 * l2 + 0.3 * l3**2 - 0.5 * np.sin(l4) + 0.4 * l1 * l2
    y = (base + 0.6 * a_0[rows] + 0.6 * b_0[cols]
         + treat * (tau + 0.4 * a_t[rows] * b_t[cols]) + rng.normal(0.0, 0.5, n))

    return {
        "L": np.column_stack([l1, l2, l3, l4, l5]),
        "A": treat,
        "Y": y,
        "g": analyst_propensity(h),
        "m1": base + tau,
        "m0": base,
        "rows": rows,
        "cols": cols,
    }


def make_row_latents(m: int, reps: int, rng: np.random.Generator) -> dict:
    """Draw m row latents, each repeated reps times consecutively.

    Args:
        m (int): number of distinct row latents.
        reps (int): consecutive repetitions per latent.
        rng (np.random.Generator): source of randomness.

    Returns:
        dict: latent arrays of length m * reps, keyed "x", "m", "t", "0".
    """
    return {k: np.repeat(rng.standard_normal(m), reps) for k in ("x", "m", "t", "0")}
