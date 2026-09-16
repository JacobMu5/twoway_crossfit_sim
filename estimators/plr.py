"""
Module for the partially linear DML estimator.

PLRDMLEstimator combines a cross-fitting design with a base learner and
estimates theta by the Robinson moment
    theta_hat = mean(y_tilde * d_tilde) / mean(d_tilde^2)
where y_tilde and d_tilde are the cross-fitted residuals. The standard
error is the additive two-way cluster-robust one, used for every design.
Because the DGP returns the true nuisances, each fit also records the
term-by-term bias dissection; the exact error decomposition behind it
is stated in protocols/P02_plr_anatomy.md.

Classes:
    PLRDMLEstimator: DML estimator with pluggable design and learner.
"""

from __future__ import annotations

import math

import numpy as np

from dgps.plr import ClusteredSample
from estimators.designs import (
    BAGGED_DESIGNS,
    DESIGNS,
    EST_SEED_OFFSET,
    Split,
    require_canonical_grid,
)
from estimators.diagnostics import error_decomposition
from estimators.learners import LEARNERS, LearnerSpec


def two_way_variance(
    psi: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n_rows: int,
    n_cols: int,
    variant: str = "cgm",
    splits: list[Split] | None = None,
) -> float:
    """Two-way score variance, scaled for sqrt(var) / (n * denom).

    "additive" pools squared row sums and squared column sums; "cgm"
    additionally subtracts the own-cell diagonal (Cameron et al. 2011)
    and is floored at zero.
    "chiang" uses Chiang et al. (2022, eq. 2.6) within the supplied test
    blocks, rescaled for sqrt(var) / (n * denom).

    Args:
        psi (np.ndarray): score values, shape (n,).
        rows, cols (np.ndarray): cluster indices, shape (n,).
        n_rows, n_cols (int): number of row and column clusters.
        variant (str): "additive", "cgm", or "chiang". Defaults to "cgm".
        splits (list[Split] | None): multiway folds; required for "chiang".

    Returns:
        float: pooled score-sum variance or the rescaled fold counterpart.
    """
    if variant == "chiang":
        if not splits:
            raise ValueError("variant='chiang' requires the multiway splits")
        gamma = 0.0
        for sp in splits:
            r, c = rows[sp.pred], cols[sp.pred]
            nr, nc = len(np.unique(r)), len(np.unique(c))
            block_sum = two_way_variance(
                psi[sp.pred], r, c, n_rows, n_cols, variant="additive",
            )
            gamma += min(nr, nc) * block_sum / (nr * nc) ** 2
        return len(psi) ** 2 * gamma / (len(splits) * min(n_rows, n_cols))
    rsum = np.bincount(rows, weights=psi, minlength=n_rows)
    csum = np.bincount(cols, weights=psi, minlength=n_cols)
    base = float(rsum @ rsum) + float(csum @ csum)
    if variant == "additive":
        return base
    return max(base - float(psi @ psi), 0.0)


def _fold_predict(
    splits: list[Split],
    x: np.ndarray,
    target: np.ndarray,
    learner: LearnerSpec,
    seed: int,
) -> np.ndarray:
    """Fit one fold learner per split and assemble out-of-fold predictions."""
    out = np.empty(len(target))
    for split_id, sp in enumerate(splits):
        model = learner.make(seed + 100003 * (split_id + 1))
        model.fit(x[sp.train], target[sp.train])
        out[sp.pred] = np.asarray(model.predict(x[sp.pred]))
    return out


class PLRDMLEstimator:
    """Partially linear DML estimator with a pluggable cross-fitting design.

    Attributes:
        design (str): key into DESIGNS or BAGGED_DESIGNS, or "oracle" for
            the true nuisances.
        learner (str): key into LEARNERS, used for the treatment nuisance.
        learner_l (str): learner for the outcome nuisance; defaults to
            `learner`. A different choice is the mixed-learner arm.
        n_bags (int | None): override for the bagged designs' bag count B;
            None uses the learner's default. Ignored by the fold designs.
        sub_exponent (float | None): override for the bagged designs'
            subsampling exponent gamma; None uses the design default.
            Ignored by the fold designs.
    """

    def __init__(
        self,
        design: str,
        learner: str,
        learner_l: str | None = None,
        n_bags: int | None = None,
        sub_exponent: float | None = None,
    ) -> None:
        if (design != "oracle" and design not in DESIGNS
                and design not in BAGGED_DESIGNS):
            raise KeyError(
                f"unknown design {design!r}; choose 'oracle' or one of "
                f"{[*DESIGNS, *BAGGED_DESIGNS]}"
            )
        if learner not in LEARNERS:
            raise KeyError(f"unknown learner {learner!r}; choose from {list(LEARNERS)}")
        if learner_l is not None and learner_l not in LEARNERS:
            raise KeyError(f"unknown learner {learner_l!r}; choose from {list(LEARNERS)}")
        self.design: str = design
        self.learner: str = learner
        self.learner_l: str = learner_l or learner
        self.n_bags: int | None = n_bags
        self.sub_exponent: float | None = sub_exponent
        self._theta_hat: float = math.nan
        self._se_hat: float = math.nan
        self._diagnostics: dict[str, float] = {}
        self._oob_stats: dict[str, float] | None = None

    @property
    def name(self) -> str:
        """Label combining design and learner(s), e.g. as_iid+gbm."""
        if self.learner_l == self.learner:
            return f"{self.design}+{self.learner}"
        return f"{self.design}+{self.learner}/{self.learner_l}"

    @property
    def theta_hat(self) -> float:
        """Point estimate from the most recent fit."""
        return self._theta_hat

    @property
    def se_hat(self) -> float:
        """Additive two-way cluster-robust standard error."""
        return self._se_hat

    @property
    def diagnostics(self) -> dict[str, float]:
        """Bias-dissection and Jacobian-anatomy terms from the last fit."""
        return self._diagnostics

    def _splits(self, sample: ClusteredSample, seed: int) -> list[Split]:
        """Build the design's fold splits from the grid dimensions and seed."""
        if len(sample.y) != sample.n_rows * sample.n_cols:
            raise ValueError(
                "fold designs assume one observation per cell of the N x M grid"
            )
        if self.design == "multiway":
            require_canonical_grid(
                sample.rows, sample.cols, sample.n_rows, sample.n_cols
            )
        rng = np.random.default_rng(seed)
        return DESIGNS[self.design](sample.n_rows, sample.n_cols, rng)

    def _nuisances(self, sample: ClusteredSample, seed: int):
        """Cross-fitted outcome and treatment predictions under the design.

        Fold designs build the splits once and share them between both
        nuisances, as in Chiang et al. (2022) (DoubleML does the same).
        The bagged designs draw a separate bag stream for the outcome
        nuisance, this is to have less covariance and joint movement of cells
        in the drawn bags and protect against avoidable bias.
        """
        self._oob_stats = None
        if self.design == "oracle":
            return sample.l0.copy(), sample.m0.copy()
        if self.design in BAGGED_DESIGNS:
            predict = BAGGED_DESIGNS[self.design]
            kwargs: dict = {} if self.n_bags is None else {"n_bags": self.n_bags}
            if self.sub_exponent is not None:
                kwargs["sub_exponent"] = self.sub_exponent
            l_hat = predict(
                sample.x, sample.y, sample.rows, sample.cols,
                sample.n_rows, sample.n_cols, LEARNERS[self.learner_l],
                seed + 7919, **kwargs,
            )
            m_hat, self._oob_stats = predict(
                sample.x, sample.d, sample.rows, sample.cols,
                sample.n_rows, sample.n_cols, LEARNERS[self.learner],
                seed, return_stats=True, **kwargs,
            )
            return l_hat, m_hat
        splits = self._splits(sample, seed)
        l_hat = _fold_predict(splits, sample.x, sample.y, LEARNERS[self.learner_l], seed)
        m_hat = _fold_predict(splits, sample.x, sample.d, LEARNERS[self.learner], seed)
        return l_hat, m_hat

    def fit(self, sample: ClusteredSample, seed: int | None = None) -> None:
        """Estimate theta and its standard error, and record the dissection.

        Args:
            sample (ClusteredSample): one realised data set.
            seed (int | None): seed for the random parts of estimation;
                offset internally so it never collides with the data
                seed. Defaults to None.
        """
        est_seed = (seed + EST_SEED_OFFSET if seed is not None
                    else int(np.random.default_rng().integers(1 << 62)))
        l_hat, m_hat = self._nuisances(sample, est_seed)

        # Robinson moment and influence function
        y_tilde = sample.y - l_hat
        d_tilde = sample.d - m_hat
        denom = float(np.mean(d_tilde * d_tilde))
        theta = float(np.mean(y_tilde * d_tilde) / denom)
        psi = (y_tilde - theta * d_tilde) * d_tilde
        n = len(sample.y)

        # Additive two-way cluster SE, applied uniformly to every design
        var_sum_add = two_way_variance(
            psi, sample.rows, sample.cols, sample.n_rows, sample.n_cols,
            variant="additive",
        )
        self._theta_hat = theta
        self._se_hat = math.sqrt(var_sum_add) / (n * denom)

        # Bias dissection from the oracle nuisances
        self._diagnostics = error_decomposition(
            sample, l_hat, m_hat, denom, psi, n, var_sum_add
        )
        if self._oob_stats is not None:
            self._diagnostics.update(
                {f"oob_{k}": val for k, val in self._oob_stats.items()}
            )

        # Recreate the same multiway splits; no learners are refitted.
        self._diagnostics["se_hat_chiang"] = math.nan
        if self.design == "multiway":
            splits = self._splits(sample, est_seed)
            if any(len(sp.pred) * len(splits) != n for sp in splits):
                raise ValueError("Chiang SE with the pooled PLR fit requires equal-sized test blocks")
            var_chiang = two_way_variance(
                psi, sample.rows, sample.cols, sample.n_rows, sample.n_cols,
                variant="chiang", splits=splits,
            )
            self._diagnostics["se_hat_chiang"] = math.sqrt(var_chiang) / (n * denom)

