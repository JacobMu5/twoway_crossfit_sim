"""
Superseded DGPs, kept for reference only, no campaign imports this
module. TwoWayPLRDGP is the diffuse DGP of the early runs and
ChiangPLRDGP (with _mix) the concentrated mixture DGP. The classes are
copies of the originals; the
paper DGP lives in dgps.reveal_plr.

Classes:
    TwoWayPLRDGP: diffuse DGP (cluster info in all five covariates).
    ChiangPLRDGP: concentrated mixture DGP (labels under reveal=True).
"""

from __future__ import annotations

import math

import numpy as np

from dgps.plr import ClusteredSample


def _m0_diffuse(x: np.ndarray) -> np.ndarray:
    """P01 treatment mean: nonlinear with an interaction."""
    return np.sin(1.4 * x[:, 0]) + 0.8 * x[:, 1] * x[:, 2] + 0.6 * np.cos(1.4 * x[:, 3])


def _g0_diffuse(x: np.ndarray) -> np.ndarray:
    """P01 baseline outcome: nonlinear with an interaction."""
    return np.cos(1.4 * x[:, 0]) + 0.7 * x[:, 2] * x[:, 4] + 0.5 * np.sin(1.4 * x[:, 1])


class TwoWayPLRDGP:
    """Diffuse DGP: cluster information in all five covariates.

    Attributes:
        theta0 (float): true coefficient (the estimand).
        cov_cluster (float): strength of the row/column effects in X.
        sd_eps (float): scale of the error term.
        sd_v (float): scale of the treatment residual; E[V^2] = sd_v^2.
    """

    def __init__(
        self,
        theta0: float = 1.0,
        cov_cluster: float = 1.5,
        sd_eps: float = 1.5,
        sd_v: float = 1.5,
    ) -> None:
        self.theta0: float = theta0
        self.cov_cluster: float = cov_cluster
        self.sd_eps: float = sd_eps
        self.sd_v: float = sd_v

    @property
    def true_theta(self) -> float:
        """True value of the estimand."""
        return self.theta0

    @property
    def name(self) -> str:
        """Label used in result tables."""
        return "two_way_plr"

    def sample(
        self, n_rows: int, n_cols: int, seed: int | None = None
    ) -> ClusteredSample:
        """Draw one two-way clustered data set.

        Args:
            n_rows (int): number of row clusters N.
            n_cols (int): number of column clusters M.
            seed (int | None): seed for reproducibility. Defaults to None.

        Returns:
            ClusteredSample: the realised data with the true nuisances.
        """
        rng = np.random.default_rng(seed)
        p = 5
        n = n_rows * n_cols
        rows = np.repeat(np.arange(n_rows), n_cols)
        cols = np.tile(np.arange(n_cols), n_rows)

        # Covariates: cell noise plus row and column effects, unit variance
        x_cell = rng.normal(size=(n, p))
        x_row = rng.normal(size=(n_rows, p))[rows]
        x_col = rng.normal(size=(n_cols, p))[cols]
        cz = self.cov_cluster
        x = (x_cell + cz * (x_row + x_col)) / math.sqrt(1.0 + 2.0 * cz * cz)

        # Treatment residual: equal-thirds row/column/cell mixture
        v_row_eff = rng.normal(size=n_rows)
        v_col_eff = rng.normal(size=n_cols)
        v_cell_eff = rng.normal(size=n)
        v = self.sd_v * (v_row_eff[rows] + v_col_eff[cols] + v_cell_eff) / math.sqrt(3.0)
        sv = self.sd_v / math.sqrt(3.0)

        # Structural error: same mixture, independent draws
        e_row_eff = rng.normal(size=n_rows)
        e_col_eff = rng.normal(size=n_cols)
        e_cell_eff = rng.normal(size=n)
        eps = self.sd_eps * (e_row_eff[rows] + e_col_eff[cols] + e_cell_eff) / math.sqrt(3.0)

        # Assemble outcome and oracle nuisances
        m0 = _m0_diffuse(x)
        g0 = _g0_diffuse(x)
        d = m0 + v
        y = self.theta0 * d + g0 + eps

        return ClusteredSample(
            x=x,
            d=d,
            y=y,
            rows=rows,
            cols=cols,
            n_rows=n_rows,
            n_cols=n_cols,
            theta0=self.theta0,
            m0=m0,
            l0=self.theta0 * m0 + g0,
            v=v,
            eps=eps,
            v_row=sv * v_row_eff[rows],
            v_col=sv * v_col_eff[cols],
            v_cell=sv * v_cell_eff,
        )


def _mix(
    row: np.ndarray, col: np.ndarray, cell: np.ndarray,
    rows: np.ndarray, cols: np.ndarray, share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row/col/cell mixture with `share` of the variance on row + col.

    Returns:
        tuple: (mixture, row part, column part, cell part), each shape (n,).
    """
    wc = math.sqrt(share / 2.0)
    we = math.sqrt(max(1.0 - share, 0.0))
    parts = (wc * row[rows], wc * col[cols], we * cell)
    return parts[0] + parts[1] + parts[2], parts[0], parts[1], parts[2]


class ChiangPLRDGP:
    """Concentrated DGP: cluster information in X1 and X2 only.

    Under reveal=True, X1 and X2 are the exact row and column labels;
    under reveal=False they are row/col/cell mixtures.

    Attributes:
        theta0 (float): true coefficient (the estimand).
        sd_v (float): scale of the treatment residual; E[V^2] = sd_v^2.
        sd_eps (float): scale of the error term.
        resid_share (float): share of residual variance on row + col.
        cov_share (float): share of covariate variance on row + col.
        reveal (bool): if True, X1 and X2 are exact row/column labels.
    """

    def __init__(
        self,
        theta0: float = 1.0,
        sd_v: float = 1.0,
        sd_eps: float = 1.0,
        resid_share: float = 2.0 / 3.0,
        cov_share: float = 2.0 / 3.0,
        reveal: bool = False,
    ) -> None:
        self.theta0: float = float(theta0)
        self.sd_v: float = float(sd_v)
        self.sd_eps: float = float(sd_eps)
        self.resid_share: float = float(resid_share)
        self.cov_share: float = float(cov_share)
        self.reveal: bool = bool(reveal)

    @property
    def true_theta(self) -> float:
        """True value of the estimand."""
        return self.theta0

    @property
    def name(self) -> str:
        """Label used in result tables."""
        return f"chiang_plr_rs{self.resid_share:g}" + ("_reveal" if self.reveal else "")

    def m0_of_x(self, x: np.ndarray) -> np.ndarray:
        """True treatment mean function E[D | X]."""
        return 0.9 * x[:, 0] - 0.6 * x[:, 1] + 0.6 * np.sin(1.5 * x[:, 2])

    def g0_of_x(self, x: np.ndarray) -> np.ndarray:
        """True baseline outcome function (outcome mean net of treatment)."""
        return 0.7 * x[:, 0] + 0.5 * np.cos(1.4 * x[:, 1]) + 0.4 * x[:, 2] - 0.3 * x[:, 3]

    def sample(
        self, n_rows: int, n_cols: int, seed: int | None = None
    ) -> ClusteredSample:
        """Draw one two-way clustered data set.

        Args:
            n_rows (int): number of row clusters N.
            n_cols (int): number of column clusters M.
            seed (int | None): seed for reproducibility. Defaults to None.

        Returns:
            ClusteredSample: the realised data with the true nuisances.
        """
        rng = np.random.default_rng(seed)
        n = n_rows * n_cols
        rows = np.repeat(np.arange(n_rows), n_cols)
        cols = np.tile(np.arange(n_cols), n_rows)

        # Covariates: cluster-loaded X1, X2 (exact labels under reveal)
        if self.reveal:
            x1 = rng.standard_normal(n_rows)[rows]
            x2 = rng.standard_normal(n_cols)[cols]
        else:
            x1 = _mix(rng.standard_normal(n_rows), rng.standard_normal(n_cols),
                      rng.standard_normal(n), rows, cols, self.cov_share)[0]
            x2 = _mix(rng.standard_normal(n_rows), rng.standard_normal(n_cols),
                      rng.standard_normal(n), rows, cols, self.cov_share)[0]
        x = np.column_stack([x1, x2, rng.standard_normal(n), rng.standard_normal(n)])

        # Residuals: shared-share mixture, independent systems
        v, v_row, v_col, v_cell = _mix(
            rng.standard_normal(n_rows), rng.standard_normal(n_cols),
            rng.standard_normal(n), rows, cols, self.resid_share,
        )
        eps = self.sd_eps * _mix(
            rng.standard_normal(n_rows), rng.standard_normal(n_cols),
            rng.standard_normal(n), rows, cols, self.resid_share,
        )[0]
        v = self.sd_v * v

        # Assemble outcome and oracle nuisances
        m0 = self.m0_of_x(x)
        g0 = self.g0_of_x(x)
        d = m0 + v
        y = self.theta0 * d + g0 + eps

        return ClusteredSample(
            x=x,
            d=d,
            y=y,
            rows=rows,
            cols=cols,
            n_rows=n_rows,
            n_cols=n_cols,
            theta0=self.theta0,
            m0=m0,
            l0=self.theta0 * m0 + g0,
            v=v,
            eps=eps,
            v_row=self.sd_v * v_row,
            v_col=self.sd_v * v_col,
            v_cell=self.sd_v * v_cell,
        )