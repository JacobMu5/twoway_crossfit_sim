"""
Module for executing a given Monte Carlo scenario.

This module contains the SimulationRunner class, which executes a
scenario by combining a DGP and an estimator. It handles data
generation, estimation, and result collection. Replication sim_id draws
data with seed = first_seed + sim_id and passes the same seed to the
estimator, so results do not depend on worker count or execution order.

Classes:
    SimulationRunner: Runs simulations and summarizes results.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import t as student_t

from sim_infrastructure.protocols import DGPProtocol, EstimatorProtocol

Z_95 = 1.959963984540054

# Diagnostics whose Monte Carlo error matters for the leak claims;
# the summary carries a <key>_mcse column for each one present.
_MCSE_KEYS = ("leak",)


class SimulationRunner:
    """Runs Monte Carlo simulations for a given DGP and estimator.

    Attributes:
        dgp (DGPProtocol): data-generating process.
        estimator (EstimatorProtocol): estimator to fit on each draw.
        records (list[dict]): one record per replication with the
            estimate, standard error, coverage, and all diagnostics.
    """

    def __init__(
        self,
        dgp: DGPProtocol,
        estimator: EstimatorProtocol,
    ) -> None:
        self.dgp: DGPProtocol = dgp
        self.estimator: EstimatorProtocol = estimator
        self.records: list[dict] = []

    def simulate(
        self,
        n_sim: int,
        n_rows: int,
        n_cols: int,
        first_seed: int | None = None,
    ) -> None:
        """Runs simulations and stores per-replication records.

        Args:
            n_sim (int): number of simulations to run.
            n_rows (int): number of row clusters per simulated data set.
            n_cols (int): number of column clusters per simulated data set.
            first_seed (int | None): starting random seed for
                reproducibility. Defaults to None.
        """
        self.records = []
        truth = self.dgp.true_theta
        t_crit = float(student_t.ppf(0.975, max(min(n_rows, n_cols) - 1, 1)))

        # Core simulation loop
        for sim_id in range(n_sim):
            seed = first_seed + sim_id if first_seed is not None else None
            # 1. Draw data
            sample = self.dgp.sample(n_rows, n_cols, seed=seed)
            # 2. Fit the estimator
            self.estimator.fit(sample, seed=seed)
            # 3. Record estimate, standard error, coverage, diagnostics
            theta_hat = self.estimator.theta_hat
            se_hat = self.estimator.se_hat
            record = {
                "sim_id": sim_id,
                "theta_hat": theta_hat,
                "se_hat": se_hat,
                "err": theta_hat - truth,
                "covered": float(abs(theta_hat - truth) <= Z_95 * se_hat),
                "covered_t": float(abs(theta_hat - truth) <= t_crit * se_hat),
            }
            record.update(self.estimator.diagnostics)
            self.records.append(record)

    def summarize_results(self) -> dict:
        """Return the inference metrics plus the mean of every diagnostic.

        The metrics follow the standing rule of the protocols (bias with
        MCSE, SD, RMSE, mean SE, coverage, and so on). Keys listed in
        _MCSE_KEYS also get a <key>_mcse column.

        Returns:
            dict: summary statistics for the completed scenario.
        """
        # Pull the record columns into arrays once
        n_rep = len(self.records)
        err = np.array([r["err"] for r in self.records])
        theta = np.array([r["theta_hat"] for r in self.records])
        se = np.array([r["se_hat"] for r in self.records])
        covered = np.array([r["covered"] for r in self.records])
        covered_t = np.array([r["covered_t"] for r in self.records])

        # Scalars that feed more than one metric
        sd = float(theta.std(ddof=1)) if n_rep > 1 else float("nan")
        bias_mcse = (float(err.std(ddof=1) / math.sqrt(n_rep))
                     if n_rep > 1 else float("nan"))
        mean_se = float(se.mean())
        coverage = float(covered.mean())
        theta_bar = theta.mean()
        bias_elim = np.array(
            [float(abs(t - theta_bar) <= Z_95 * s) for t, s in zip(theta, se)]
        )

        summary = {
            "reps": n_rep,
            "bias": float(err.mean()),
            "bias_mcse": bias_mcse,
            "sd": sd,
            "rmse": float(math.sqrt(np.mean(err**2))),
            "mean_se": mean_se,
            "se_ratio": float(mean_se / sd) if sd > 0 else float("nan"),
            "coverage": coverage,
            "coverage_mcse": float(math.sqrt(coverage * (1 - coverage) / n_rep)),
            "coverage_t": float(covered_t.mean()),
            "bias_elim_coverage": float(bias_elim.mean()),
        }

        # Average every diagnostic key present in any record, keeping
        # first-seen order
        core = {"sim_id", "theta_hat", "se_hat", "err", "covered", "covered_t"}
        diag_keys: list[str] = []
        for record in self.records:
            for key in record:
                if key not in core and key not in summary and key not in diag_keys:
                    diag_keys.append(key)
        for key in diag_keys:
            vals = np.array([r[key] for r in self.records if key in r])
            summary[key] = float(vals.mean())
            if key in _MCSE_KEYS and len(vals) > 1:
                summary[f"{key}_mcse"] = float(vals.std(ddof=1) / math.sqrt(len(vals)))
        return summary

