"""
Module for the paper DGP: the concentrated reveal PLR data-generating
process. RevealPLRDGP puts the exact row/column labels in X1/X2 (so a
flexible learner can absorb V's cluster shocks which produces the as-IID leak) 
and uses the diffuse DGP's (can be found in the leagacy_plr.py code)
nonlinear/interaction nuisance forms with Cov(g0, m0) = 0. 
Superseded DGPs live in dgps.legacy_plr.

Classes:
    RevealPLRDGP: the paper DGP (revealed labels, orthogonal nuisances).
"""

from __future__ import annotations

import math

import numpy as np

from dgps.plr import ClusteredSample


class RevealPLRDGP:
    """Concentrated two-way PLR DGP: X1 and X2 ARE the cluster labels.

    X1 is the row effect, X2 the column effect (one N(0,1) draw per
    cluster), X3-X5 are pure cell noise, and the residuals place
    resid_share of their variance on row + column (equal thirds
    row/col/cell at the default 2/3). The nuisance functions use the
    diffuse DGP's nonlinear/interaction forms with Cov(g0, m0) = 0.

    Attributes:
        theta0 (float): true coefficient (the estimand).
        sd_v (float): scale of the treatment residual; for
            resid_share in [0, 1], E[V^2] = sd_v^2.
        sd_eps (float): scale of the error term.
        resid_share (float): share of residual variance on row + col.
    """

    def __init__(
        self,
        theta0: float = 1.0,
        sd_v: float = 1.0,
        sd_eps: float = 1.0,
        resid_share: float = 2.0 / 3.0,
    ) -> None:
        self.theta0: float = float(theta0)
        self.sd_v: float = float(sd_v)
        self.sd_eps: float = float(sd_eps)
        self.resid_share: float = float(resid_share)

    @property
    def true_theta(self) -> float:
        """True value of the estimand."""
        return self.theta0

    @property
    def name(self) -> str:
        """Label used in result tables.

        Fresh label for the combo nuisances: do not confuse 
        with the archived chiang_CSVs 
        """
        return "combo_reveal"

    def m0_of_x(self, x: np.ndarray) -> np.ndarray:
        """True treatment mean function E[D | X]."""
        return (np.sin(1.4 * x[:, 0]) + 0.8 * x[:, 1] * x[:, 2]
                + 0.6 * np.cos(1.4 * x[:, 3]))

    def g0_of_x(self, x: np.ndarray) -> np.ndarray:
        """True baseline outcome (orthogonal to m0: Cov(g0, m0) = 0)."""
        return (np.cos(1.4 * x[:, 0]) + 0.7 * x[:, 2] * x[:, 4]
                + 0.5 * np.sin(1.4 * x[:, 1]))

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

        # X1/X2 are the exact row/column labels; X3-X5 pure cell noise
        x1 = rng.standard_normal(n_rows)[rows]
        x2 = rng.standard_normal(n_cols)[cols]
        x = np.column_stack([x1, x2, rng.standard_normal(n),
                             rng.standard_normal(n), rng.standard_normal(n)])

        # Residuals: resid_share of the variance on row + column.
        wc = math.sqrt(self.resid_share / 2.0)
        we = math.sqrt(max(1.0 - self.resid_share, 0.0))
        v_row = wc * rng.standard_normal(n_rows)[rows]
        v_col = wc * rng.standard_normal(n_cols)[cols]
        v_cell = we * rng.standard_normal(n)
        v = self.sd_v * (v_row + v_col + v_cell)
        eps = self.sd_eps * (wc * rng.standard_normal(n_rows)[rows]
                             + wc * rng.standard_normal(n_cols)[cols]
                             + we * rng.standard_normal(n))

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