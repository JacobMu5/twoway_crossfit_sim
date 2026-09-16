"""Monte Carlo loop for one DGP-estimator pair."""

import math

import numpy as np
from scipy.stats import t as student_t

from sim_infrastructure.protocols import DGPProtocol, EstimatorProtocol

Z_95 = 1.959963984540054

_MCSE_KEYS = ("leak",)
_CORE_KEYS = {"sim_id", "theta_hat", "se_hat", "err", "covered", "covered_t", "covered_chiang"}


class SimulationRunner:
    def __init__(self, dgp: DGPProtocol, estimator: EstimatorProtocol) -> None:
        self.dgp = dgp
        self.estimator = estimator
        self.records: list[dict] = []

    def simulate(
        self,
        n_sim: int,
        n_rows: int,
        n_cols: int,
        first_seed: int | None = None,
    ) -> None:
        self.records = []
        truth = self.dgp.true_theta
        t_crit = float(student_t.ppf(0.975, max(min(n_rows, n_cols) - 1, 1)))

        for sim_id in range(n_sim):
            seed = first_seed + sim_id if first_seed is not None else None
            sample = self.dgp.sample(n_rows, n_cols, seed=seed)
            self.estimator.fit(sample, seed=seed)
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
            se_ch = record.get("se_hat_chiang", float("nan"))
            record["covered_chiang"] = (
                float(abs(theta_hat - truth) <= Z_95 * se_ch)
                if se_ch == se_ch else float("nan")
            )            
            self.records.append(record)

    def summarize_results(self) -> dict:
        n_rep = len(self.records)
        err = np.array([r["err"] for r in self.records])
        theta = np.array([r["theta_hat"] for r in self.records])
        se = np.array([r["se_hat"] for r in self.records])
        covered = np.array([r["covered"] for r in self.records])
        covered_t = np.array([r["covered_t"] for r in self.records])
        covered_chiang = np.array([r.get("covered_chiang", float("nan"))
                                   for r in self.records])

        sd = float(theta.std(ddof=1)) if n_rep > 1 else float("nan")
        bias_mcse = (float(err.std(ddof=1) / math.sqrt(n_rep))
                     if n_rep > 1 else float("nan"))
        mean_se = float(se.mean())
        coverage = float(covered.mean())
        theta_bar = theta.mean()
        bias_elim = np.abs(theta - theta_bar) <= Z_95 * se

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
            "coverage_chiang": (float(np.nanmean(covered_chiang))
                                if np.isfinite(covered_chiang).any() else float("nan")),            
            "bias_elim_coverage": float(bias_elim.mean()),
        }

        for key in (k for k in self.records[0] if k not in _CORE_KEYS):
            vals = np.array([r[key] for r in self.records])
            summary[key] = float(vals.mean())
            if key in _MCSE_KEYS and len(vals) > 1:
                summary[f"{key}_mcse"] = float(vals.std(ddof=1) / math.sqrt(len(vals)))
        summary["se_ratio_chiang"] = (
            float(summary.get("se_hat_chiang", float("nan")) / sd)
            if sd > 0 else float("nan")
        )
        return summary

