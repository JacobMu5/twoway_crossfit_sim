"""Scalar PLIV score with three nuisances; reuse PLR training and variance code."""
 
from __future__ import annotations
 
import math
 
import numpy as np
 
from dgps.pliv import PLIVSample
from estimators.designs import BAGGED_DESIGNS, DESIGNS, EST_SEED_OFFSET, Split
from estimators.learners import LEARNERS
from estimators.plr import _fold_predict, two_way_variance
 
Z_95 = 1.959963984540054
_STREAM = {"y": 7919, "d": 0, "z": 15838}  # independent bag stream per nuisance
 
 
class PLIVDMLEstimator:
    """PLIV DML estimator with a pluggable design and one shared base learner."""
 
    def __init__(self, design, learner, n_bags=None, sub_exponent=None):
        if (design != "oracle" and design not in DESIGNS
                and design not in BAGGED_DESIGNS):
            raise KeyError(f"unknown design {design!r}")
        if learner not in LEARNERS:
            raise KeyError(f"unknown learner {learner!r}")
        self.design, self.learner = design, learner
        self.n_bags, self.sub_exponent = n_bags, sub_exponent
        self._theta_hat = self._se_hat = math.nan
        self._diagnostics: dict[str, float] = {}
 
    @property
    def name(self):
        return f"{self.design}+{self.learner}"
 
    @property
    def theta_hat(self):
        return self._theta_hat
 
    @property
    def se_hat(self):
        return self._se_hat
 
    @property
    def diagnostics(self):
        return self._diagnostics
 
    def _predict(self, sample, target, truth, stream, seed, splits):
        """Cross-fitted prediction of one target under the chosen design."""
        if self.design == "oracle":
            return truth.copy()
        learner = LEARNERS[self.learner]
        if self.design in BAGGED_DESIGNS:
            kw = {} if self.n_bags is None else {"n_bags": self.n_bags}
            if self.sub_exponent is not None:
                kw["sub_exponent"] = self.sub_exponent
            return BAGGED_DESIGNS[self.design](
                sample.x, target, sample.rows, sample.cols, sample.n_rows,
                sample.n_cols, learner, seed + _STREAM[stream], **kw)
        return _fold_predict(splits, sample.x, target, learner, seed)
 
    def fit(self, sample: PLIVSample, seed=None):
        """Estimate theta and its standard error from the instrumental moment."""
        est_seed = (seed + EST_SEED_OFFSET if seed is not None
                    else int(np.random.default_rng().integers(1 << 62)))
        n = len(sample.y)
        # Keep all units of a cell in the same training/evaluation fold.
        one_per_cell = n == sample.n_rows * sample.n_cols
        cells = None if one_per_cell else sample.rows * sample.n_cols + sample.cols
        splits = None
        if self.design in DESIGNS:
            cell_splits = DESIGNS[self.design](
                sample.n_rows, sample.n_cols, np.random.default_rng(est_seed))
            splits = cell_splits if one_per_cell else [
                Split(np.flatnonzero(np.isin(cells, s.train)),
                      np.flatnonzero(np.isin(cells, s.pred))) for s in cell_splits]
 
        l_hat = self._predict(sample, sample.y, sample.mean_y, "y", est_seed, splits)
        m_hat = self._predict(sample, sample.d, sample.mean_d, "d", est_seed, splits)
        pi_hat = self._predict(sample, sample.z, sample.mean_z, "z", est_seed, splits)
 
        y_t, d_t, z_t = sample.y - l_hat, sample.d - m_hat, sample.z - pi_hat
        jac = float(np.mean(d_t * z_t))
        num = float(np.mean(y_t * z_t))
        ok = np.isfinite(jac) and jac != 0.0 and np.isfinite(num)
        theta = num / jac if ok else math.nan
        psi = (y_t - theta * d_t) * z_t
        var_add = two_way_variance(psi, sample.rows, sample.cols,
                                   sample.n_rows, sample.n_cols, variant="additive")
        se_add = math.sqrt(var_add) / (n * abs(jac)) if ok else math.nan
        # CGM subtracts squared cell totals, including within-cell dependence.
        own = (float(psi @ psi) if one_per_cell
               else float(np.bincount(cells, weights=psi)
                          @ np.bincount(cells, weights=psi)))
        se_cgm = (math.sqrt(max(var_add - own, 0.0)) / (n * abs(jac))
                  if ok else math.nan)
 
        def _mse(hat, truth):  # oracle means are optional for application data
            return float(np.mean((hat - truth) ** 2)) if truth is not None else math.nan
 
        self._theta_hat, self._se_hat = theta, se_add
        self._diagnostics = {
            "se_hat_add": se_add, "se_hat_cgm": se_cgm, "jacobian": jac,
            "mse_y": _mse(l_hat, sample.mean_y),
            "mse_d": _mse(m_hat, sample.mean_d),
            "mse_z": _mse(pi_hat, sample.mean_z),
            "estimation_failure": float(not ok),
            "covered_cgm": (float(abs(theta - sample.theta0) <= Z_95 * se_cgm)
                            if ok else math.nan),
        }