"""Side check for Section 5.3: repeat the two-way partition S = 10 times on the lead_plr samples.

The S fits of a sample are combined in two ways: the mean or the median of the S estimates, with
the standard error of Chernozhukov et al. (2018, Section 3.4), or the mean of the S predictions of
each cell before one pooled moment. Partition 0 is the partition of the lead_plr two-way run.

    python supplementary/repeated_partitions.py   ->  results/final2000/repeated_partitions_{records,summary}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from estimators.designs import EST_SEED_OFFSET  
from estimators.diagnostics import error_decomposition  
from estimators.plr import two_way_variance  
from sim_infrastructure.runner import Z_95  
from sim_infrastructure.scenarios import RUNS  

# the two-way scenario of lead_plr: same DGP, grid, learner and seeds
SC = next(sc for sc in RUNS["lead_plr"] if sc.design == "multiway" and sc.learner == "gbm"
          and sc.dgp_params.get("include_signatures", True))
S, STRIDE = 10, 1_000_003


def moment(s, l_hat, m_hat):
    """Robinson estimate with the additive SE and the error terms, as in PLRDMLEstimator.fit."""
    d, y = s.d - m_hat, s.y - l_hat
    denom = np.mean(d * d)
    theta = np.mean(d * y) / denom
    psi = (y - theta * d) * d
    var = two_way_variance(psi, s.rows, s.cols, s.n_rows, s.n_cols, variant="additive")
    return {"theta_hat": theta, **error_decomposition(s, l_hat, m_hat, denom, psi, len(psi), var)}


def replication(i):
    seed = SC.first_seed + i
    s = SC.dgp(**SC.dgp_params).sample(SC.n_rows, SC.n_cols, seed=seed)
    est = SC.estimator(**SC.estimator_params)
    with threadpool_limits(limits=1):
        fits = [est._nuisances(s, seed + EST_SEED_OFFSET + k * STRIDE) for k in range(S)]
    one = pd.DataFrame([moment(s, l_hat, m_hat) for l_hat, m_hat in fits])
    rows = []
    for name, agg in (("mean", np.mean), ("median", np.median)):       
        theta = agg(one.theta_hat)
        spread = (one.theta_hat - theta) ** 2
        rows.append({"design": f"rep_{name}_S{S}", "theta_hat": theta, "Eb2": one.Eb2.mean(),
                     "se_hat_add": np.sqrt(agg(one.se_hat_add ** 2 + spread)),
                     "se_hat_unadj": np.sqrt(agg(one.se_hat_add ** 2))})
    for k in (1, 2, 5, 10):                                           
        l_bar, m_bar = np.mean(fits[:k], axis=0)
        rows.append({"design": f"rep_prediction_S{k}", **moment(s, l_bar, m_bar)})
    return [{"sim_id": i, "err": r["theta_hat"] - s.theta0, **r} for r in rows]


def summarize(records):
    """One line per way of combining, with the column names of the campaign summaries."""
    unadjusted = records[records.design == "rep_mean_S10"].assign(      
        design="rep_mean_S10_unadjusted", se_hat_add=lambda d: d.se_hat_unadj)  
    rows = []
    for design, g in pd.concat([records, unadjusted]).groupby("design"):
        e, se = g.err, g.se_hat_add
        rows.append({"design": design, "reps": len(g), "bias": e.mean(), "sd": g.theta_hat.std(),
                     "rmse": np.sqrt((e**2).mean()), "mean_se": se.mean(),
                     "coverage": (e.abs() <= Z_95 * se).mean(),
                     "bias_elim_coverage": ((e - e.mean()).abs() <= Z_95 * se).mean(), "Eb2": g.Eb2.mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    runs = Parallel(n_jobs=-1)(delayed(replication)(i) for i in range(SC.n_simulations))
    records = pd.DataFrame([row for run in runs for row in run])
    folder = ROOT / "results" / "final2000"
    records.to_csv(folder / "repeated_partitions_records.csv", index=False)
    summarize(records).to_csv(folder / "repeated_partitions_summary.csv", index=False)
    print(summarize(records).round(3))
